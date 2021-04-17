import pandas as pd
import sklearn.svm as svm
import pickle
import yaml


x_train = pd.read_csv("/Users/prahalad/Desktop/dvc/Data/Cleansed/xTrainingSet.csv")
y_train = pd.read_csv("/Users/prahalad/Desktop/dvc/Data/Cleansed/yTrainingSet.csv")

x_test = pd.read_csv("/Users/prahalad/Desktop/dvc/Data/Cleansed/xTestingSet.csv")

params = yaml.safe_load(open('params.yaml'))['train']


svmModel = svm.SVC(kernel=params['kernal'])
svmModel.fit(x_train,y_train)


with open("/Users/prahalad/Desktop/dvc/Data/Model/model.pkl",'wb') as fd:
    pickle.dump(svmModel,fd)
