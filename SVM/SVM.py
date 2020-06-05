import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer=datasets.load_breast_cancer()
# Features
print(cancer.feature_names)
# Labels
print(cancer.target_names)

# Splitting Data
x = cancer.data  # All of the features
y = cancer.target  # All of the labels

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

# First five instances
print(x_train[:5], y_train[:5])
classes=['malignant', 'benign']

#Clasifier
clf= svm.SVC(kernel="linear", C=2)
#Worst, many variance depending of n_neighbors
#clf=KNeighborsClassifier(n_neighbors=13)
clf.fit(x_train, y_train)
# Predict
y_pred=clf.predict(x_test)
# Accuracy
acc=metrics.accuracy_score(y_test, y_pred)

print(acc)