import pandas as pd
import sklearn.linear_model
import pickle
import tensorflow as tf

x_train = pd.read_csv("/Users/prahalad/Desktop/dvc/Data/Cleansed/xTrainingSet.csv")
y_train = pd.read_csv("/Users/prahalad/Desktop/dvc/Data/Cleansed/yTrainingSet.csv")

x_test = pd.read_csv("/Users/prahalad/Desktop/dvc/Data/Cleansed/xTestingSet.csv")


import sklearn.svm as svm

svmModel = svm.SVC()
svmModel.fit(x_train,y_train)


with open("/Users/prahalad/Desktop/dvc/Data/Model/model.pkl",'wb') as fd:
    pickle.dump(svmModel,fd)
