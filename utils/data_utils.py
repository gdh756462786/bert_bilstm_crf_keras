"""
################################data_utils.py################################
程序名称:     data_utils.py
功能描述:     数据处理
创建人名:     wuxinhui
创建日期:     2019-07-31
版本说明:     v1.0
################################data_utils.py################################
"""

import os
import tqdm
import json
import pickle
import random
import numpy as np
from sklearn.model_selection import train_test_split

class Data_utils(object):

	"""docstring for Data_utils"""

	def __init__(self):
		super(Data_utils, self).__init__()
		self._min_count = 1
		self._sample_file = "../data/sample.txt"
		self._vocab = {}
		self._tag = {}
		self._sample = []
	
	def _reader(self):
		fid = open(self._sample_file, "r", encoding="utf-8")
		senl = []
		tagl = []
		for line in fid.readlines():
			try:
				sen,tag = line.strip().split("\t")
				senl.append(self._strQ2B(sen).lower())
				tagl.append(tag)
			except:
				self._sample.append(("".join(senl),tagl))
				senl = []
				tagl = []
		fid.close()
		return

	def _strQ2B(self, ustring):
		rstring = ''
		for uchar in ustring:
			inside_code = ord(uchar)
			if inside_code == 12288:
				inside_code = 32
			elif (inside_code >= 65281 and inside_code <= 65374):
				inside_code -= 65248
			rstring += chr(inside_code)
		return rstring

	def _one_hot(self, labels, dim):
		results = np.zeros((len(labels), dim))
		for i, label in enumerate(labels):
			results[i][label] = 1
		return results
	
	def _vocab_build(self):
		tags = []
		for l in self._sample:
			senl, tagl = l
			tags.append(tagl)
		tags = sum(tags, [])

		self._tag = {t:tags.count(t) for t in set(tags)}
		self._id2tag = {i+1:j for i,j in enumerate(self._tag)}
		self._tag2id = {v:k for k,v in self._id2tag.items()}

		pickle.dump((self._tag,self._id2tag,self._tag2id), open("../saved_models/tag_dict.pl", "wb"), -1)
		return

	def _tag2id_func(self, tagl):
		return [self._tag2id[t] for t in tagl]

	def _id2tag_func(self, ids):
		return [self._id2tag[i] for i in ids]

	def _sample_build(self, shuffle=True):
		y = []
		X = []
		for line in self._sample:
			senl,tagl = line
			X.append(senl)
			y.append(tagl)
		Xtrain, Xtest, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
		pickle.dump((Xtrain,y_train), open("../data/train.pl", "wb"), -1)
		pickle.dump((Xtest,y_test), open("../data/test.pl", "wb"), -1)
		return
		
if __name__ == "__main__":

	dhelp = Data_utils()
	dhelp._reader()
	dhelp._vocab_build()	
	dhelp._sample_build()