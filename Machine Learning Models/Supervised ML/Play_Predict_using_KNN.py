import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def CheckAccuracy(target_test,prediction):
    result = accuracy_score(target_test,prediction)
    return result * 100

def PlayPredictor():
    border = "*"*50
    dataset = pd.read_csv('PlayPredictor.csv')

    df_col = list(dataset.columns)
    print(border)
    print("Feature names of the given dataset:- ")
    print([df_col[0:2]])
    print(border)

    print("Target of the given dataset:- ")
    print([df_col[2]])
    print(border)

    le = preprocessing.LabelEncoder()

    data_encoded = dataset[df_col[0:3]].apply(le.fit_transform)
    data = data_encoded[df_col[0:2]]
    target = data_encoded[df_col[2]]

    print("Total encoded dataset is as follows:- ")
    print(data_encoded)
    print(border)

    #Step 3 - Use entire data for training
    classifier = KNeighborsClassifier(n_neighbors=3)

    classifier.fit(data,target)
    result = classifier.predict([[2,1],[1,0]])
    print("The prediction of KNN using entire dataset for training and with K = 3 is:- \n",result)
    print(border)

    data_train,data_test,target_train,target_test = train_test_split(data,target,test_size=0.5)
    
    kobj = KNeighborsClassifier(n_neighbors=2)
    kobj.fit(data_train,target_train)
    prediction = classifier.predict(data_test)
    print("Result of prediction using KNN algorithm is :- \n ",prediction)
    print(border)

    Accuracy = CheckAccuracy(target_test,prediction)
    return Accuracy
    

def main():
    result = PlayPredictor()
    print("Accuracy of the Machine Learning Algorithm is:- \n",result)
    
    
if __name__ == "__main__":
    main()