import pandas as pd
import sklearn.linear_model
import pickle


x_train = pd.read_csv("/Users/prahalad/Desktop/dvc/Data/Cleansed/xTrainingSet.csv")
y_train = pd.read_csv("/Users/prahalad/Desktop/dvc/Data/Cleansed/yTrainingSet.csv")

y_test = pd.read_csv("/Users/prahalad/Desktop/dvc/Data/Cleansed/xTestingSet.csv")

with open("/Users/prahalad/Desktop/dvc/Data/Model/model.pkl",'rb') as f:
    svmModel = pickle.load(f)

accuracy = svmModel.score(x_train,y_train)

predictedOutput_svm = svmModel.predict(y_test)


pd.DataFrame(pd.Series([accuracy],index=['accuracy'])).to_json(path_or_buf="/Users/prahalad/Desktop/dvc/Data/Evaluation/result.json")
