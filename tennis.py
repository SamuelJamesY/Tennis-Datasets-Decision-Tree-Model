'''
Building a Decision Tree Classifier based off Gini function
'''

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier 
from sklearn.tree import export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

def load_data():
	'''
	Load the data and seperate into features and targets
	'''
	df = pd.read_csv('tennis.csv')
	print(df.head())
	X = df.iloc[:,1:-1]
	y = df.iloc[:,-1]
	return X,y

def build_dummy_variables(X,y):
	'''
	All the options inside a column are uniquely encoded (Huffman Tree)
	'''
	dummy_cols = [ 'outlook', 'temp', 'humidity', 'wind'] # the variables to uniquely encode in feature fraame
	ff = pd.get_dummies(X,columns=dummy_cols,drop_first=True)
	names = list(ff.columns)
	features = ff.to_numpy() # convert features to numpy array
	tf = pd.get_dummies(y,columns=['play'],drop_first=True) # the variable to uniquely encode in target frame
	targets = tf.to_numpy() # convert targets to numpy array
	targets = targets.flatten() # turn targets from 2d to 1d
	return features,targets,names

def train_test_split_data(features,targets):
	'''
	Split Data into Targets and features
	'''
	xtrain,xtest,ytrain,ytest = train_test_split(features,targets,test_size=0.4,random_state=25)
	return xtrain,xtest,ytrain,ytest

def build_decision_tree(xtrain,xtest,ytrain,ytest,names):
	# set max features to auto, most features to consider when splitting the data in the tree 
	# minimum samples required for a leaf node to exist is 1
	# the max depth is the maximum depth a tree can be
	# build a decision tree 
	dt = DecisionTreeClassifier(max_features='auto',min_samples_leaf=1,max_depth=20,random_state=25)
	dtree = dt.fit(xtrain,ytrain) # fit the decision tree to the training data
	# weights mean the amount of don't play and will play, feature names are names of all the dummy columns
	d_rules = export_text(dtree,show_weights=True,feature_names=names)
	print(d_rules) # rules of the decision tree
	# with our decision tree we can predict whether we play or not from the test set
	ypred = dtree.predict(xtest)
	acc = accuracy_score(ytest,ypred)
	print(acc,'  Accuracy Score')
	cm = confusion_matrix(ytest,ypred) # y axis is actual value, x axis is predicted value
	print(cm, ' Confusion Matrix')
	class_names = ['No', 'Yes']
	dfcm = pd.DataFrame(cm, index=class_names, columns=class_names)
	return dfcm

def cm_heatmap(dfcm):
	'''
	Plot a heatmap of whether we classified playing or not playing tennis correctly or incorrectly
	'''
	plt.figure(figsize=(5, 4))
	sns.heatmap(dfcm, annot=True,  cmap="Blues", fmt=".0f")
	plt.title("Confusion Matrix")
	plt.ylabel("True Class")
	plt.xlabel("Predicted Class")
	plt.savefig('confusion_matrix.png',bbox_inches="tight")
	plt.clf()

def main():
	X,y = load_data()
	features,targets,names = build_dummy_variables(X,y)
	xtrain,xtest,ytrain,ytest = train_test_split_data(features,targets)
	dfcm = build_decision_tree(xtrain,xtest,ytrain,ytest,names)
	cm_heatmap(dfcm)

if __name__ == '__main__':
	main()