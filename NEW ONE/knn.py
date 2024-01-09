from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import datasets

iris = datasets.load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.30)
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

# Calculate accuracy and print correct and wrong predictions
accuracy = accuracy_score(y_test, y_pred)
print('Correct predictions:', accuracy)
print('Wrong predictions:', 1 - accuracy)