from utils import db_connect
engine = db_connect()

# your code here
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn import tree
from sklearn.metrics import accuracy_score
from pickle import dump

datos = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv")

datos = datos.drop_duplicates().reset_index(drop = True)

X = datos.drop("Outcome", axis = 1)
y = datos["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

selection_model = SelectKBest(k = 7)
selection_model.fit(X_train, y_train)

selected_columns = X_train.columns[selection_model.get_support()]
X_train_sel = pd.DataFrame(selection_model.transform(X_train), columns = selected_columns)
X_test_sel = pd.DataFrame(selection_model.transform(X_test), columns = selected_columns)

X_train_sel["Outcome"] = y_train.values
X_test_sel["Outcome"] = y_test.values
X_train_sel.to_csv("data/processed/datos_limpios_train.csv", index = False)
X_test_sel.to_csv("data/processed/datos_limpios_test.csv", index = False)

datos_train = pd.read_csv("data/processed/datos_limpios_train.csv")
datos_test = pd.read_csv("data/processed/datos_limpios_test.csv")

plt.figure(figsize=(12, 6))
pd.plotting.parallel_coordinates(datos, "Outcome", color = ("#E58139", "#39E581", "#8139E5"))
plt.show()

X_train = datos_train.drop(["Outcome"], axis = 1)
y_train = datos_train["Outcome"]
X_test = datos_test.drop(["Outcome"], axis = 1)
y_test = datos_test["Outcome"]

modelado = DecisionTreeClassifier(random_state = 42)
modelado.fit(X_train, y_train)

fig = plt.figure(figsize=(15,15))
tree.plot_tree(modelado, feature_names = list(X_train.columns), class_names = ["0", "1", "2"], filled = True)
plt.show()

y_pred = modelado.predict(X_test)

accuracy_score(y_test, y_pred)

hyperparams = {
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid = GridSearchCV(modelado, hyperparams, scoring = "accuracy", cv = 10)

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

grid.fit(X_train, y_train)

print(f"Best hyperparameters: {grid.best_params_}")

modelado = DecisionTreeClassifier(criterion = "entropy", max_depth = 5, min_samples_leaf = 4, min_samples_split = 2, random_state = 42)
modelado.fit(X_train, y_train)

y_pred = modelado.predict(X_test)

accuracy_score(y_test, y_pred)

dump(modelado, open("models/tree_classifier.sav", "wb"))