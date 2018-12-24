# -*- coding: utf-8 -*-
#!/usr/bin/python
import codecs
import os
import math
import sys
import pandas as pd 
import numpy as np 
from pyvi import ViTokenizer
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier as knn
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

class text_classification:
	model = None
	modelknn = None
	train_data = None
	test_data = None
	label_data = None
	label_test = None
	attribute = []
	idf = {}
	check = 0
	def Load_Idf(self):
		feature= []
		Temp = pd.read_csv("/home/cse/Work/Dataset/idf.csv")
		for i in Temp:
			feature.append(i)
			if self.check == 0:
				self.attribute.append(i)
		self.check = 1
		for i in feature:
			self.idf[i] = Temp[i]
		return

	def Train_Data(self):
		feature = []
		Temp = pd.read_csv("/home/cse/Work/Dataset/idf.csv")
		for i in Temp:
			if self.check == 0:
				self.attribute.append(i)
		self.check = 1
		data = pd.read_csv("/home/cse/Work/Dataset/data.csv")
		for i in data:
			feature.append(i)
		feature = feature[:-1]
		X = data[feature]
		Y = data['label']

		self.train_data, self.test_data, self.label_data, self.label_test = train_test_split(X,Y,test_size=0.3,random_state=1)
		print("Training model......")
		self.model = svm.LinearSVC(random_state=0)
		self.model.fit(self.train_data,self.label_data)

		self.modelknn = knn(n_neighbors=7)
		self.modelknn.fit(self.train_data,self.label_data)

		joblib.dump(self.model,"/home/cse/Work/Dataset/MODEL.pkl")
		joblib.dump(self.modelknn,"/home/cse/Work/Dataset/MODELKNN.pkl")
		print("Finish training.")
		print()
		return

	def Validation(self):
		label = []
		for i in self.label_test:
			label.append(i)
		print("SVM Validation .....")
		validate = self.model.predict(self.test_data)
		count = 0
		for i in range(len(validate)):
			if validate[i] == label[i]:
				count += 1
		print("SVM Validation Accuracy: {} %".format(count/len(label) * 100))

		labelknn = []
		for i in self.label_test:
			labelknn.append(i)
		print("KNN Validation .....")
		validateknn = self.modelknn.predict(self.test_data)
		countknn = 0
		for i in range(len(validateknn)):
			if validateknn[i] == labelknn[i]:
				countknn += 1
		print("KNN Validation Accuracy: {} %".format(countknn/len(labelknn) * 100))
		return

	def Convert_Document(self,file):
		document = codecs.open(file,encoding="utf8", errors='ignore').read()
		document = ViTokenizer.tokenize(document)
		document = document.lower()
		print()
		print("=================Document after run ViTokenizer=============")
		print(document)
		table = str.maketrans("\"\'“”+=[]1234567890!?.,-/@#$%^&*()\\{}<>:;'", 41*" ")
		document = document.translate(table)
		document = document.split(" ")
		return self.Vectorization(document)

	def Vectorization(self,document):
		tf = {}
		vector = []
		for word in self.attribute:
			tf[word] = 0
		for word in document:
			if word in self.attribute:
				tf[word] += 1
		for word in self.attribute:
			tmp = tf[word] * self.idf[word]
			vector.append(np.asscalar(tmp.values))
		print()
		print("=================Document after Vectorization =============")
		print(vector)
		return vector

	def Predict(self,model,file):
		self.Load_Idf()
		pre = joblib.load(model)
		Vector = self.Convert_Document(file)
		Tmp = pre.predict([Vector])
		print()
		print("=================Predict result=============")
		if Tmp == 1:
			print("Cong dong")
		elif Tmp == 2:
			print("Du lich")
		elif Tmp == 3:
			print("Gia dinh")
		elif Tmp == 4:
			print("Giao duc")
		elif Tmp == 5:
			print("Giai tri")
		elif Tmp == 6:
			print("Khoa hoc")
		elif Tmp == 7:
			print("Kinh doanh")
		elif Tmp == 8:
			print("Phap luat")
		elif Tmp == 9:
			print("So hoa")
		elif Tmp == 10:
			print("Suc khoe")
		elif Tmp == 11:
			print("Tam su")
		elif Tmp == 12:
			print("The gioi")
		elif Tmp == 13:
			print("The thao")
		elif Tmp == 14:
			print("Thoi su")
		elif Tmp == 15:
			print("Xe")
		return

	def Test(self,f):
		sample_test = []
		label_target = []
		self.Load_Idf()
		docs = os.listdir(f)
		for d in docs:
			file = f + "/" + d
			if d[0:2] == 'co':
				label_target.append(1)
			elif d[0:2] == 'du':
				label_target.append(2)
			elif d[0:5] == 'gia_d':
				label_target.append(3)
			elif d[0:4] == 'giao':
				label_target.append(4)
			elif d[0:4] == 'giai':
				label_target.append(5)
			elif d[0:2] == 'kh':
				label_target.append(6)
			elif d[0:2] == 'ki':
				label_target.append(7)
			elif d[0:2] == 'ph':
				label_target.append(8)
			elif d[0:2] == 'so':
				label_target.append(9)
			elif d[0:2] == 'su':
				label_target.append(10)
			elif d[0:2] == 'ta':
				label_target.append(11)
			elif d[0:5] == 'the_g':
				label_target.append(12)
			elif d[0:5] == 'the_t':
				label_target.append(13)
			elif d[0:4] == 'thoi':
				label_target.append(14)
			elif d[0:2] == 'xe':
				label_target.append(15)
			sample_test.append(self.Convert_Document(file))
		print("SVM Testing Model....")
		result = []
		count = 0
		result = self.model.predict(sample_test)
		for i in range(len(result)):
			if result[i] == label_target[i]:
				count += 1
		print("SVM Test Accuracy: {} %".format(count/len(result) * 100))

		print("KNN Testing Model....")
		resultknn = []
		countknn = 0
		resultknn = self.modelknn.predict(sample_test)
		for i in range(len(resultknn)):
			if resultknn[i] == label_target[i]:
				countknn += 1
		print("KNN Test Accuracy: {} %".format(countknn/len(result) * 100))
		
		return


if __name__ == '__main__':
	classifier = text_classification()
	classifier.Train_Data()
	classifier.Validation()
	#classifier.Test("/home/cse/Work/Dataset/Test")
	print()
	print("=======================SVM predict new File ======================")
	classifier.Predict("/home/cse/Work/Dataset/MODEL.pkl","/home/cse/Work/Dataset/cong_dong(148).txt")
	print("=======================KNN predict new File======================")
	classifier.Predict("/home/cse/Work/Dataset/MODELKNN.pkl","/home/cse/Work/Dataset/cong_dong(148).txt")