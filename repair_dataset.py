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
import csv



class make_csv:
	dictionary = {}
	documents_string = []
	documents_string_test = []
	documents_number = []
	features = []
	nW = 0 # so luong cac tu khac nhau
	time_appear = {}
	count_docs = {}
	idf = {}
	samples = []
	labels = []
	def Convert_String_To_Number(self,document):
		doc_number = ""
		for i in range(len(document)-1):
			doc_number += str(self.dictionary[document[i]]) + " "
		doc_number += str(self.dictionary[document[len(document)-1]])
		return doc_number

	def IDF(self,f):
		for word in self.features:
			self.count_docs[word] = 0

		for word in self.features:
			for doc in self.documents_string:
				if word in doc:
					self.count_docs[word] += 1
		for word in self.features:
			self.idf[word] = math.log(self.nW/self.count_docs[word])
		with open(f,'w') as file:
			for i in range(len(self.features)-1):
				file.write(self.features[i])
				file.write(',')
			file.write(self.features[len(self.features)-1])
			file.write('\n')
			for i in range(len(self.features)-1):
				file.write(str(self.idf[self.features[i]]))
				file.write(',')
			file.write(str(self.idf[self.features[len(self.features)-1]]))
			file.close()
		return

	def TF(self,document):
		tf = {}
		for word in self.features:
			tf[word] = 0
		for word in document:
			if word in self.features:
				tf[word] += 1
		return tf

	def TF_IDF(self):
		for doc in self.documents_string:
			tfidf = []
			tf = self.TF(doc)
			for word in self.features:
				tfidf.append(tf[word] * self.idf[word])
			self.samples.append(tfidf)
		return
	#tien xu li
	def Pretreatment(self,document): 
		document = ViTokenizer.tokenize(document)
		document = document.lower()
		table = str.maketrans("\"\'“”+=[]1234567890!?.,-/@#$%^&*()\\{}<>:;'", 41*" ")
		document = document.translate(table)
		document = document.split(" ")
		self.Add_Dictionary(document)
		self.documents_string.append(document)
		#self.documents_number.append(convert_string_to_number(document))
		return

	def Add_Dictionary(self,document):

		for word in document:
			if word not in self.dictionary:
				self.dictionary[word] = self.nW
				self.time_appear[word] = 1
				self.nW += 1
			else:
				self.time_appear[word] += 1
		return	
	def Remove_Stopword(self):
		for word in self.dictionary:
			if self.time_appear[word] >=60 and self.time_appear[word] <=600:
				self.features.append(word)
		return

	def Access_Data(self,f):
		docs = os.listdir(f)
		for d in docs:
			file = f + d
			print(file)
			if d[0:2] == 'co':
				self.labels.append(1)
			elif d[0:2] == 'du':
				self.labels.append(2)
			elif d[0:5] == 'gia_d':
				self.labels.append(3)
			elif d[0:4] == 'giao':
				self.labels.append(4)
			elif d[0:4] == 'giai':
				self.labels.append(5)
			elif d[0:2] == 'kh':
				self.labels.append(6)
			elif d[0:2] == 'ki':
				self.labels.append(7)
			elif d[0:2] == 'ph':
				self.labels.append(8)
			elif d[0:2] == 'so':
				self.labels.append(9)
			elif d[0:2] == 'su':
				self.labels.append(10)
			elif d[0:2] == 'ta':
				self.labels.append(11)
			elif d[0:5] == 'the_g':
				self.labels.append(12)
			elif d[0:5] == 'the_t':
				self.labels.append(13)
			elif d[0:4] == 'thoi':
				self.labels.append(14)
			elif d[0:2] == 'xe':
				self.labels.append(15)	
			content = codecs.open(file,encoding="utf8", errors='ignore').read()
			#tien xu li tung noi dung bai viet
			self.Pretreatment(content)
		return

	#chia thu muc data thanh 2 thu muc la data va test
	def Repare(self,f,data_link,test_link):
		dirs = os.listdir(f)
		for dir in dirs:
			forder = f + "/" + dir
			docs = os.listdir(forder)
			i = 0
			for doc in docs:
				file = forder + "/" + doc
				content = codecs.open(file,encoding="utf8", errors='ignore').read()
				filename = ""
				if i < 550:
					filename = data_link + dir + str(i) + ".txt"
				else:
					filename = test_link + dir + str(i) + ".txt"
				print(filename)
				file = open(filename,"w")
				file.write(content)
				i += 1
				file.close()
		return
	def Make_File_Data(self,f):
		with open(f,'w') as file:
			for i in range(len(self.features)):
				file.write(self.features[i])
				file.write(',')
			file.write('label')

			file.write('\n')
			for i in range(len(self.samples)):
				for j in range(len(self.samples[i])):
					file.write(str(self.samples[i][j]))
					file.write(',')
				file.write(str(self.labels[i]))
				file.write('\n')
		file.close()
		return
if __name__ == '__main__':
	creater = make_csv()
	creater.Repare("/home/cse/Work/Dataset/Temp","/home/cse/Work/Dataset/Train/","/home/cse/Work/Dataset/Test/")
	creater.Access_Data("/home/cse/Work/Dataset/Train/")
	creater.Remove_Stopword()
	print("size of features = {}".format(len(creater.features)))
	print("number of different words = {}".format(creater.nW))
	creater.IDF('/home/cse/Work/Dataset/idf.csv')
	creater.TF_IDF()
	creater.Make_File_Data('/home/cse/Work/Dataset/data.csv')