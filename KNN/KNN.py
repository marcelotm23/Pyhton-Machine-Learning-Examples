from sklearn import preprocessing
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
print(data.head())

# Preprocesing convert string to numbers
le=preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

X = list(zip(buying, maint, door, persons, lug_boot, safety))  # features
y = list(cls)  # labels

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)
print(x_train,y_test)

model = KNeighborsClassifier(n_neighbors=9)
#Train
model.fit(x_train, y_train)
#Accuaricity
acc=model.score(x_test,y_test)
print(acc)
#Predict
predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]
for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
    # Now we will we see the neighbors of each point in our testing data
    n = model.kneighbors([x_test[x]], 9, True)
    print("N: ", n)