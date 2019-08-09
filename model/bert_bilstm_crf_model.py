"""
################################bert_bilstm_crf_model.py################################
程序名称:     bert_bilstm_crf_model.py
功能描述:     BERT预训练模型+双向长短时记忆网络+条件随机场
创建人名:     wuxinhui
创建日期:     2019-08-03
版本说明:     v1.0
################################bert_bilstm_crf_model.py################################
"""

import numpy as np
import os, sys, datetime, pickle
import pickle
import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras.utils import plot_model
from keras.engine.topology import Layer
from keras.initializers import Constant
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy
from keras.preprocessing.sequence import pad_sequences
from keras_bert import load_trained_model_from_checkpoint
from keras_bert import Tokenizer
from keras_bert.layers import TokenEmbedding
from keras_pos_embd import PositionEmbedding
from keras_layer_normalization import LayerNormalization
from keras_multi_head import MultiHeadAttention
from keras import regularizers
from keras import backend as K

class bert_bilstm_crf_model(object):

	"""docstring for bert_bilstm_crf_model"""

	def __init__(self, args):
		super(bert_bilstm_crf_model, self).__init__()
		self.embeds_dim = args.embeds_dim
		self.embeds_dir = args.embeds_dir
		self.dict_dir = args.dict_dir
		self.pretrain_model_dir = args.pretrain_model_dir
		self.layerid = args.layerid
		self.max_seq_len = args.max_seq_len
		self.batch_size = args.batch_size
		self.epochs = args.epochs
		self.saved_models_dir = args.saved_models_dir
		self.layer_dict = [7, 15, 23, 31, 39, 47, 55, 63, 71, 79, 87, 95, 103]
		self._loader_dict_()
		self.model = self._model_compile_()

	def _loader_dict_(self):
		self.vocab_dict = {}
		with open(os.path.join(self.pretrain_model_dir,"vocab.txt"),"r",encoding="utf-8") as fid:
			for line in fid.readlines():
				self.vocab_dict[line.strip()] = len(self.vocab_dict)
		tag_dict = pickle.load(open(os.path.join(self.dict_dir, "tag_dict.pl"), 'rb'))
		self._tag, self._id2tag, self._tag2id = tag_dict
		return
		
	def _model_compile_(self):
		layerN = 12
		bert_model =load_trained_model_from_checkpoint(
							os.path.join(self.pretrain_model_dir,"bert_config.json"),
							os.path.join(self.pretrain_model_dir,"bert_model.ckpt"),
							seq_len=self.max_seq_len,
							training=False,
							use_adapter=True,
							trainable=['Encoder-{}-MultiHeadSelfAttention-Adapter'.format(i+1) for i in range(layerN)]+
							['Encoder-{}-FeedForward-Adapter'.format(i+1) for i in range(layerN)]+
							['Encoder-{}-MultiHeadSelfAttention-Norm'.format(i+1) for i in range(layerN)]+
							['Encoder-{}-FeedForward-Norm'.format(i+1) for i in range(layerN)]
			)
		bert_out = bert_model.get_layer(index=self.layer_dict[self.layerid]).output
		l2_reg = regularizers.l2(1e-8)
		trans_sens = Bidirectional(LSTM(
								self.embeds_dim,
								return_sequences=True,
								kernel_regularizer=l2_reg))(bert_out)
		crf_tags = CRF(len(self._tag), sparse_target=True)(trans_sens)
		model = Model(bert_model.inputs, crf_tags)
		model.summary()
		model.compile(
					optimizer=Adam(lr=0.001),
					loss=crf_loss,
					metrics =[crf_accuracy]
			)
		plot_model(model, to_file=os.path.join(self.saved_models_dir,'bert_bilstm_crf_model.png'), show_shapes=True)
		return model
		
	def _model_train_(self, train_text):
		X, y = train_text
		input_ids, input_mask, input_type = self._text_process(X)
		trainy = self._label_encoder(y, one_hot=False)
		callbacks =[
				ReduceLROnPlateau(),
				ModelCheckpoint(filepath=os.path.join(self.saved_models_dir, "bert_bilstm_crf_model.h5"), \
								save_best_only=True)
			]
		self.model.fit(x=[input_ids,input_mask], 
					y=trainy, 
					batch_size=self.batch_size, 
					epochs=self.epochs, 
					verbose=1, 
					callbacks=callbacks,
					validation_split=0.1,  
					shuffle=True)
		return

	def _model_test_(self, test_text):
		X, y = test_text
		input_ids, input_mask, input_type = self._text_process(X)
		testy = self._label_encoder(y, one_hot=False)
		print(y[0])
		print(testy[0])
		self.model.load_weights(os.path.join(self.saved_models_dir, "bert_bilstm_crf_model.h5"))
		score = self.model.evaluate(x=[input_ids,input_mask], y=testy, verbose=1)
		print("the test loss for model: %s" %score[0])
		print("the test accuracy for model: %s" %score[1])
		return
		
	def _model_predict_(self, talk):
		input_ids, input_mask, input_type = self._text_process([talk])
		self.model.load_weights(os.path.join(self.saved_models_dir, "bert_bilstm_crf_model.h5"))
		tag = self.model.predict([input_ids,input_mask])[0][1:len(talk)+1]
		tag = self._id2tag_func(self._label_decoder(tag))
		return tag

	def _text_process(self, text):
		Tokener = Tokenizer(self.vocab_dict)
		encoder = [Tokener.encode(first=doc, max_len=self.max_seq_len) for doc in text]
		input_ids = [i[0] for i in encoder]
		input_type = [i[1] for i in encoder]
		input_mask = [[0 if l==0 else 1 for l in i] for i in input_ids]
		return (input_ids,input_mask,input_type)
		
	def _char2id_func(self, senl):
		return [self._char2id.get(s,1) for s in senl]

	def _id2char_func(self, ids):
		return [self._id2char[i] for i in ids]

	def _tag2id_func(self, tagl):
		return [self._tag2id[t] for t in tagl]

	def _id2tag_func(self, ids):
		return [self._id2tag.get(i,"U") for i in ids]
		
	def _label_decoder(self, tag):
		return [np.argmax(ids) for ids in tag]

	def _label_encoder(self, labels, one_hot=False):
		tags = [[0]+[self._tag2id[t] for t in l]+[0] for l in labels]
		padding_tag = pad_sequences(tags, self.max_seq_len, padding="post", truncating="post")
		if one_hot == True:
			res_tag = np.eye(len(self._tag), dtype='float32')[padding_tag]
		else:
			res_tag = np.expand_dims(padding_tag, 2)
		return res_tag

