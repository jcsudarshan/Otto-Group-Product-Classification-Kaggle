import time
import preprocess

from sklearn.neighbors import KNeighborsClassifier


def loaddata():
####Data set is loaded into the train.csv file
print "loading data..."
data,label = preprocess.loadTrainSet()

####Divide the loaded data set into train data and validation data
val_data = data[0:6000]
val_label = label[0:6000]
train_data = data[6000:]
train_label = label[6000:]
test_data = preprocess.loadTestSet()
####The data is loaded into test.csv file
return train_data,train_label,val_data,val_label,test_data



def knn(train_data,train_label,val_data,val_label,test_data,name = "knn_submission.csv"):
print "Start training KNN Classifier..."
####validation set is evaluated
knnClf = KNeighborsClassifier(n_neighbors=20)
	knnClf.fit(train_data,train_label)
	
	val_pred_label = knnClf.predict_proba(val_data)
	logloss = preprocess.evaluation(val_label,val_pred_label)
	print "logloss of validation set:",logloss
####Classifying the set
	print "Start classify test set..."
	test_label = knnClf.predict_proba(test_data)
	preprocess.saveResult(test_label,filename = name)



if __name__ == "__main__":
	t1 = time.time()
	train_data,train_label,val_data,val_label,test_data = loaddata()
	knn(train_data,train_label,val_data,val_label,test_data) 
	t2 = time.time()
	print "Done! It cost",t2-t1,"s"
