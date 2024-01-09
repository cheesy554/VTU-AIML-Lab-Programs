from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn import datasets

# To load the dataset manually
# iris = pd.read_csv('Naive_Bayes/iris.csv')
# data = np.array(iris.iloc[:,0:-1])
# target = np.array(iris.iloc[:,-1])

iris = datasets.load_iris()
data = iris.data
target = iris.target
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.30)

# Create a Naive Bayes classifier
clf = GaussianNB()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
