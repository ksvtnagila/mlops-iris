from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import tree
from datetime import datetime

clf = GaussianNB()
decision_tree = tree.DecisionTreeClassifier(criterion='gini')
timeNow = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")

classes = {
    0: "Iris Setosa",
    1: "Iris Versicolour",
    2: "Iris Virginica"
}

def load_model():
	X, y = datasets.load_iris(return_X_y=True)

	X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
	clf.fit(X_train, y_train)

	acc = accuracy_score(y_test, clf.predict(X_test))
	print(f"Model trained with accuracy: {round(acc, 3)} Time Stamp: ",timeNow)
    


def load_model2():
    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
    

    #Train DT based on scaled training set
    decision_tree.fit(X_train, y_train)

    #Print performance
    print('Time Stamp: ',timeNow)
    print('The accuracy of the Decision Tree classifier on training data is {:.2f}'.format(decision_tree.score(X_train, y_train)))
    print('The accuracy of the Decision Tree classifier on test data is {:.2f}'.format(decision_tree.score(X_test, y_test)))

def predict(query_data):
	x = list(query_data.dict().values())
	prediction = clf.predict([x])[0] 
	print(f"Model prediction: {classes[prediction]}")
	return classes[prediction]

def predict2(query_data):
    x = list(query_data.dict().values())
    prediction=decision_tree.predict([x])[0]
    print(f"Model prediction: {classes[prediction]}")
    return classes[prediction]
	




