"""
################################run_ner.py################################
程序名称:     run_ner.py
功能描述:     实体识别
创建人名:     wuxinhui
创建日期:     2019-07-31
版本说明:     v1.0
################################run_ner.py################################
"""

import numpy as np
import os
import argparse
import time
import random
import sys
import pickle

sys.path.append("./model")
sys.path.append("./utils")

from model.bert_bilstm_crf_model import bert_bilstm_crf_model
from utils.data_utils import Data_utils


class NER(object):

	"""docstring for NER"""

	def __init__(self):
		super(NER, self).__init__()

	def run(self):
		parser = argparse.ArgumentParser(description='HAN model for multi-classes Classifier')
		parser.add_argument('--train_data', type=str, default='data', help='train data source')
		parser.add_argument('--test_data', type=str, default='data', help='test data source')
		parser.add_argument('--dict_dir', type=str, default='saved_models', help='dir for dict')
		parser.add_argument('--saved_models_dir', type=str, default='saved_models', help='dir for model saved')
		parser.add_argument('--pretrain_model_dir', type=str, default="pretrain_model/chinese_L-12_H-768_A-12", help='dir for pretrain model')
		parser.add_argument('--layerid', type=int, default=12, help='layer for bert model')
		parser.add_argument('--embeds_dir', type=str, default=None, help='test data source')
		parser.add_argument('--embeds_dim', type=int, default=50, help='dim of embedding')
		parser.add_argument('--batch_size', type=int, default=64, help='sample of each minibatch')
		parser.add_argument('--epochs', type=int, default=5, help='epoch of training')
		parser.add_argument('--max_seq_len', type=int, default=20, help='max length for sentence')
		args = parser.parse_args()

		train_data = pickle.load(open(os.path.join(args.train_data, "train.pl"), "rb"))
		test_data = pickle.load(open(os.path.join(args.test_data, "test.pl"), "rb"))

		model = bert_bilstm_crf_model(args)
		model._model_train_(train_data)
		model._model_test_(test_data)
		
		seq = "你好"
		tag = model._model_predict_(seq)
		print(tag)
		
if __name__ == "__main__":
	
	NER().run()