#-*- coding:utf8 -*-#
"""
Created on 2015/05/23
Kaggle competition "Otto Group Product Classification Challenge". 

@author:Arima_ram

"""
import csv
import random
import numpy  as np



def loadTrainSet():
#####Training sets are loaded
	traindata = []
	trainlabel = []
	table = {"Class_1":1,"Class_2":2,"Class_3":3,"Class_4":4,"Class_5":5,"Class_6":6,"Class_7":7,"Class_8":8,"Class_9":9}
	with open("train.csv") as f:
		rows = csv.reader(f)
		rows.next()
		for row in rows:
			l = []
			for i in range(1,94):
				l.append(int(row[i]))
			traindata.append(l)
			trainlabel.append(table.get(row[-1]))
	f.close()

	traindata = np.array(traindata,dtype="float")
	trainlabel = np.array(trainlabel,dtype="int")
	#Standardize(zero-mean,normalization)
	mean = traindata.mean(axis=0)
	std = traindata.std(axis=0)
	traindata = (traindata - mean)/std
	
	#The data are shuffled in the following code segments
	randomIndex = [i for i in xrange(len(trainlabel))]
	random.shuffle(randomIndex)
	traindata = traindata[randomIndex]
	trainlabel = trainlabel[randomIndex]
	return traindata,trainlabel


def loadTestSet():
#Test sets are loaded in this function
	testdata = []
	with open("test.csv") as f:
		rows = csv.reader(f)
		rows.next()
		for row in rows:
			l = []
			for i in range(1,94):
				l.append(int(row[i]))
			testdata.append(l)
	f.close()
	testdata = np.array(testdata,dtype="float")
	#Standardize(zero-mean,nomalization)
	mean = testdata.mean(axis=0)
	std = testdata.std(axis=0)
	testdata = (testdata - mean)/std
	return testdata


#Function evaluation performed as per the contest rules
def evaluation(label,pred_label):
	num = len(label)
	logloss = 0.0
	for i in range(num):
		p = max(min(pred_label[i][label[i]-1],1-10**(-15)),10**(-15))
		logloss += np.log(p)
	logloss = -1*logloss/num
	return logloss


#The results are saved into the submission.csv file
def saveResult(testlabel,filename = "submission.csv"):
	with open(filename,'wb') as myFile:
		myWriter=csv.writer(myFile)
		myWriter.writerow(["id","Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9"])
		id_num = 1
		for eachlabel in testlabel:
			l = []
			l.append(id_num)
			l.extend(eachlabel)
			myWriter.writerow(l)
			id_num += 1


