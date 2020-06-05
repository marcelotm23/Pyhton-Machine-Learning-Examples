import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style


data=pd.read_csv("student-mat.csv", sep=";")

print(data.head())

data = data[["G1","G2","G3","studytime","failures","absences"]]

predict = "G3"
# Features
X = np.array(data.drop([predict], 1))
# Output Labels
Y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
"""
# Find the best model
best=0
for _ in range(30):

    # Split data into testing and training data (90% to train, 10% to test)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)

    # We will start by defining the model which we will be using.
    linear= linear_model.LinearRegression()

    # Next we will train and score our model using the arrays we created in the previous tutorial.
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test) # acc stands for accuracy
    # View accuracy
    print(acc)
    # If the current model has a better score than one we've already trained then save it
    if acc > best:
        best = acc
        # Save model
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
"""
# Loading the created model
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)


# Viewing The Constants
print('Coefficient: \n', linear.coef_) # These are each slope value
print('Intercept: \n', linear.intercept_) # This is the intercept

# Predicting on Specific Students
predictions = linear.predict(x_test) # Gets a list of all predictions

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

# Beutify results showing them in plot
p = "failures" # Change this to G1, G2, studytime or absences to see other graphs
style.use("ggplot")
pyplot.scatter(data[p],data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()