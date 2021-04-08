import pandas as pd
import sklearn.linear_model
import numpy as np
import pickle


trainingSet = pd.read_csv("/Users/prahalad/Desktop/dvc-backup/Data/train.csv")
testingSet = pd.read_csv("/Users/prahalad/Desktop/dvc-backup/Data/test.csv")

testingBackup = testingSet.copy()


testingBackup['Age'] = testingBackup['Age'].fillna(value=testingBackup['Age'].mean())
testingSet['Age'] = testingSet['Age'].fillna(value=testingSet['Age'].mean())

testingBackup['Fare'] = testingBackup['Fare'].fillna(value=testingBackup[testingBackup.Pclass == 3]['Fare'].mean())
testingSet['Fare'] = testingSet['Fare'].fillna(value=testingSet[testingSet.Pclass == 3]['Fare'].mean())


trainingSet.drop(['Cabin','Ticket','Name','PassengerId'], axis=1,inplace=True)
trainingSet.dropna(inplace=True)

testingSet.drop(['Cabin','Ticket','Name','PassengerId'], axis=1,inplace=True)


genderDummies = pd.get_dummies(trainingSet['Sex'], prefix="Gender")
trainingSet = pd.concat([trainingSet,genderDummies],axis=1)
trainingSet.drop(["Sex"],axis=1,inplace=True)

genderDummies = pd.get_dummies(testingSet['Sex'], prefix="Gender")
testingSet = pd.concat([testingSet,genderDummies],axis=1)
testingSet.drop(["Sex"],axis=1,inplace=True)

trainingSet  = pd.concat( [trainingSet, pd.get_dummies(trainingSet["Embarked"], prefix="Location")],axis=1)
trainingSet.drop(["Embarked"],axis=1,inplace=True)

testingSet  = pd.concat( [testingSet, pd.get_dummies(testingSet["Embarked"], prefix="Location")],axis=1)
testingSet.drop(["Embarked"],axis=1,inplace=True)


y_train = trainingSet.pop("Survived")
print(y_train.dtypes)

x_train = trainingSet
print(x_train.dtypes)
x_test = testingSet


x_train.to_csv("/Users/prahalad/Desktop/dvc/Data/Cleansed/xTrainingSet.csv",index=False)
y_train.to_csv("/Users/prahalad/Desktop/dvc/Data/Cleansed/yTrainingSet.csv",index=False)

x_test.to_csv("/Users/prahalad/Desktop/dvc/Data/Cleansed/xTestingSet.csv",index=False)