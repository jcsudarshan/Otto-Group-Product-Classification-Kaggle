Source Code file Description

1. Dependent libraries / Software
		->sklearn
		->numpy
		
2. Data Preprocessing 
		->preprocess.py

The preprocessing work, defines the following common functions: 
		->a function of the training data set is loaded loadTrainSet (), 
		->loading function loadTestSet test data set (), 
		->the function evaluation assessment model logloss value (), and
		->to generate submissions submission.csv function saveResult ().

After data normalization, zero equalization.

3. Category

	KNN
	K-nearest neighbor algorithm, the effect is poor. Time-consuming and expensive memory. k = 20s, logloss about 1. 

	RandomForest.py
	Random Forests, n_estimators = 400s when, logloss about 0.55.

	ExtraTrees.py
	No parameter adjustment.

	GradientBoosting.py
	No parameter adjustment.

	Adaboost.py
	No parameter adjustment.

4. Implementation

	Download train.csv, test.csv from the official website, and preprocess.py, RandomForest.py placed in the same directory, you can run directly RandomForest.py.


	
	
	
	
Contest Description:

Classify products into the correct category

Get started on this competition through Kaggle Scripts

The Otto Group is one of the world’s biggest e-commerce companies, with subsidiaries in more than 20 countries, including Crate & Barrel (USA), Otto.de (Germany) and 3 Suisses (France). We are selling millions of products worldwide every day, with several thousand products being added to our product line.

A consistent analysis of the performance of our products is crucial. However, due to our diverse global infrastructure, many identical products get classified differently. Therefore, the quality of our product analysis depends heavily on the ability to accurately cluster similar products. The better the classification, the more insights we can generate about our product range.

2nd iteration

For this competition, we have provided a dataset with 93 features for more than 200,000 products. The objective is to build a predictive model which is able to distinguish between our main product categories. The winning models will be open sourced.

