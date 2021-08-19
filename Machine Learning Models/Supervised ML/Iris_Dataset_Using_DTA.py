from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
#accuracy_score is imp is testing your algo 
def MarvellousDecision():
    dataset = load_iris()

    data = dataset.data
    target = dataset.target

    print("Feature name of Iris Data set:- ")
    print(dataset.feature_names)

    print("Labels of Iris Data set:- ")
    print(dataset.target_names)

    print("First 10 data elements from iris data set are :-")

    for i in range(len(data)):
        print("ID: {}, Label:   {}, Feature:    {}".format(i,data[i],target[i]))
    #bifurcting the dataset into taining data and testing data using train_test_split
    data_train,data_test,target_train,target_test = train_test_split(data,target,train_size = 75)

    #Creating the object of the decision tree classifier algorithm
    cobj = tree.DecisionTreeClassifier()

    cobj.fit(data_train,target_train)

    ouput = cobj.predict(data_test)

    Accuracy = accuracy_score(target_test,ouput)
    return Accuracy

def main():
    ret = MarvellousDecision()
    print("Accuracy of decision tree algorithm is:- ",ret*100,"%")


if __name__ == "__main__":
    main()