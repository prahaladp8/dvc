import pandas as pd
import sklearn.linear_model
import pickle

x_train = pd.read_csv("/Users/prahalad/Desktop/dvc/Data/Cleansed/xTrainingSet.csv")
y_train = pd.read_csv("/Users/prahalad/Desktop/dvc/Data/Cleansed/yTrainingSet.csv")

x_test = pd.read_csv("/Users/prahalad/Desktop/dvc/Data/Cleansed/xTestingSet.csv")

clf = sklearn.linear_model.LogisticRegression()
clf.fit(x_train,y_train)

predictedOutput_lr = clf.predict(x_test)
predictedOutput_lr = pd.Series(predictedOutput_lr,name='Survived')


with open("/Users/prahalad/Desktop/dvc/Data/Model/model.pkl",'wb') as fd:
    pickle.dump(clf,fd)

'''
clf.predict(x_train)

accuracy = clf.score(x_train,y_train)
accuracy_json_data = pd.DataFrame(pd.Series([accuracy],index=['accuracy'])).to_json(path_or_buf="/Users/prahalad/Desktop/dvc/Data/Evaluation")
'''