import pandas as pd
import sklearn.linear_model

trainingSet = pd.read_csv("/Users/prahalad/Desktop/dvc/Data/train.csv")
testingSet = pd.read_csv("/Users/prahalad/Desktop/dvc/Data/test.csv")

trainingSet.drop(['Cabin','Ticket','Name','PassengerId'], axis=1,inplace=True)
trainingSet.dropna(inplace=True)

testingSet.drop(['Cabin','Ticket','Name','PassengerId'], axis=1,inplace=True)
testingSet.dropna(inplace=True)

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

#print(trainingSet.dtypes)

y_train = trainingSet.pop("Survived")
x_train = trainingSet

#y_test = testingSet.pop("Survived")
#x_test = testingSet

clf = sklearn.linear_model.LogisticRegression()
clf.fit(x_train,y_train)
clf.predict(x_train)
print("Training Accuracy : {}".format(clf.score(x_train,y_train)))

