'''
A decision tree for a tennis dataset. Some of the predictor variables are continous numerical values.
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
	Load the data and give dummy variables for the categorical values
	Seperate the predictor and response variables and targets and features
	Also return names of dummy variables. 
	'''
	df = pd.read_csv('tennis_num.csv')
	print(df.head())
	ff = df.iloc[:,1:-1]
	dummy_cols = ['outlook','wind']
	features = pd.get_dummies(ff,columns=dummy_cols,drop_first=True)
	names = list(features.columns)
	features = features.to_numpy()
	tf = df.iloc[:,-1]
	targets = pd.get_dummies(tf,columns=['play'],drop_first=True)
	targets = targets.to_numpy()
	targets = targets.flatten()
	return features,targets,names

def train_test_split_data(features,targets):
	'''
	Split the data with a train test split of 75/25
	'''
	xtrain,xtest,ytrain,ytest = train_test_split(features,targets,test_size=0.25,random_state=42)
	return xtrain,xtest,ytrain,ytest

def decision_tree(xtrain,xtest,ytrain,ytest,names):
	'''
	Use decision tree classifier with a maximum depth of 42 and the minimum samples for a leaf of 1
	'''
	dt = DecisionTreeClassifier(max_features='auto',min_samples_leaf=1,max_depth=42,random_state=42)
	dtree = dt.fit(xtrain,ytrain)
	d_rules = export_text(dtree,feature_names=names,show_weights=True)
	print(d_rules)
	ypred = dtree.predict(xtest)
	acc = accuracy_score(ytest,ypred)
	print(acc,'  Accuracy Score')
	cm = confusion_matrix(ytest,ypred)
	class_names = ['No','Yes']
	cmdf = pd.DataFrame(cm,columns=class_names,index=class_names)
	return cmdf

def cm_heatmap(cmdf):
	'''
	Form a heatmap of correct and incorrectly classified samples using seaborn
	'''
	plt.figure(figsize=(5, 4))
	sns.heatmap(cmdf, annot=True,  cmap="Blues", fmt=".0f")
	plt.title("Confusion Matrix")
	plt.ylabel("True Class")
	plt.xlabel("Predicted Class")
	plt.savefig('confusion_matrix.png',bbox_inches="tight")
	plt.clf()

def main():
	features,targets,names = load_data()
	xtrain,xtest,ytrain,ytest = train_test_split_data(features,targets)
	cmdf = decision_tree(xtrain,xtest,ytrain,ytest,names)
	cm_heatmap(cmdf)

if __name__ == '__main__':
	main()