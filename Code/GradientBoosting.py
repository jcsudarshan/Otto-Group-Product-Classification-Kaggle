import time
import preprocess

from sklearn.ensemble import GradientBoostingClassifier

def loaddata():
#####The data set is loaded into train.csv file 
	print "loading data..."
	data,label = preprocess.loadTrainSet()
	#####The data set is divided into train data and validation data
	val_data = data[0:6000]
	val_label = label[0:6000]
	train_data = data[6000:]
	train_label = label[6000:]
	######The data is loaded into test.csv file	
	test_data = preprocess.loadTestSet()
	return train_data,train_label,val_data,val_label,test_data


#Gradient Tree Boosting
#"warm_start" support:If set as True, the solution is reused from the previous instantiation to fit and provide more estimators to the ensemble,
#otherwise, the previous solution should be erased

def gb(train_data,train_label,val_data,val_label,test_data,name="GradientBoosting_submission.csv"):
	print "start training GradientBoosting..."
	gbClf = GradientBoostingClassifier()       # params: by default
	gbClf.fit(train_data,train_label)
	######validation set is evaluated
	val_pred_label = gbClf.predict_proba(val_data)
	logloss = preprocess.evaluation(val_label,val_pred_label)
	print "logloss of validation set:",logloss
	
	######Classification of the data set
	print "Start classify test set..."
	test_label = gbClf.predict_proba(test_data)
	preprocess.saveResult(test_label,filename = name)

  

if __name__ == "__main__":
	t1 = time.time()
	train_data,train_label,val_data,val_label,test_data = loaddata()
	gb(train_data,train_label,val_data,val_label,test_data) 
	t2 = time.time()
	print "Done! It cost",t2-t1,"s"
	
	
