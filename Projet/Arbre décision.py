#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-12-04T14:04:05.352Z
"""

# Importation des bibliothèques
import	numpy	as	np
import	pandas	as	pd
import sys
np.set_printoptions(threshold=sys.maxsize)
import	warnings
import	matplotlib.pyplot	as	plt
warnings.filterwarnings('ignore')

# Lecture des données brut
df = pd.read_csv("data.csv", sep=';')
# Suppression de 3 lignes contenant des NaN
df = df.dropna()
# Isolation de la colonne "Gender" pour séparer les H et les F
X = df.drop("Target", axis=1).to_numpy()
y = df["Target"].to_numpy()

# Vérification des données
X.shape, y.shape, df.hist

# Séparation des données pour le train et le test (80% / 20%)
from sklearn.model_selection import train_test_split

# X = variables explicatives (features)
# y = variable à prédire (target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vérification des données dans le train et dans le test
X_train.shape, X_test.shape

# Classification par arbre de décision
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X,y)

# Création de l'arbre de décision
from sklearn.tree import plot_tree

plt.figure(figsize=(25,8))
plot_tree(tree, filled=True)
plt.show()

# Prédiction + précision
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report

# Prédictions
y_pred = tree.predict(X_test)

# Évaluation
print("Évaluation en moyenne :")
print("Exactitude :", accuracy_score(y_test, y_pred))
print("Précision :", precision_score(y_test, y_pred, average='weighted'))
print("Rappel :", recall_score(y_test, y_pred, average='weighted'))
print("F1 score :", f1_score(y_test, y_pred, average='weighted'))
print(classification_report(y_test, y_pred))

# Create Decision Tree classifer object
tree2 = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
tree2 = tree2.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = tree2.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",accuracy_score(y_test, y_pred))


# Create Decision Tree classifer object
tree3 = DecisionTreeClassifier(splitter="random", max_depth=3)

# Train Decision Tree Classifer
tree3 = tree3.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = tree3.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",accuracy_score(y_test, y_pred))

# https://www.datacamp.com/fr/tutorial/decision-tree-classification-python

# First, you need to install GraphViz on your system
# For Ubuntu/Debian: !apt-get install graphviz
# For Windows: Download and install from https://graphviz.org/download/
# For macOS: !brew install graphviz
# Then install pydotplus if not already installed
# !pip install pydotplus

from six import StringIO 
from IPython.display import Image  
from sklearn.tree import export_graphviz
import graphviz
import pydotplus
import os

# Set the path to GraphViz (adjust this path based on your installation)
# For Windows, it might be something like:
# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'

feature_columns = df.columns[:-1]
dot_data = StringIO()
export_graphviz(tree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_columns,class_names=['Dropout','Enrolled','Graduated'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('academics.png')
Image(graph.create_png())

# Utilisation de la détection d'anomalie d'après la matrice de confusion de la forêt aléatoire
from sklearn.ensemble import IsolationForest

# Suppression de 3 lignes contenant des NaN
df = df.dropna()

model = IsolationForest(contamination=0.01)
model.fit(df)

df.tail()

decision = model.decision_function(df)
print(decision)

predict = model.predict(df)
print(predict)

predict_series = pd.Series(predict)
predict_series.value_counts()

# Visualisation des résultats
plt.figure(figsize=(10, 5))

# Points normaux
normal_index = df.index[predict == 1]
normal_scores = decision[predict == 1]
plt.scatter(normal_index, normal_scores, label='Normal')

# Points anormaux
anomaly_index = df.index[predict == -1]
anomaly_scores = decision[predict == -1]
plt.scatter(anomaly_index, anomaly_scores, label='Anomalies')

plt.xlabel("Instance")
plt.ylabel("Score d'anomalies")
plt.legend()
plt.show()

# Comparaison entre les notes du premier et du second semestre en se basant sur les anomalies
col1 = "Curricular units 1st sem (grade)"
col2 = "Curricular units 2nd sem (grade)"

# Séparer les points normaux et anormaux
normal = df[predict == 1]
anomalies = df[predict == -1]

plt.figure(figsize=(7,5))

# Points normaux
plt.scatter(normal[col1], normal[col2], label="Normal")

# Points anormaux
plt.scatter(anomalies[col1], anomalies[col2], label="Anomalie")

plt.xlabel(col1)
plt.ylabel(col2)
plt.title("Isolation Forest sur les notes (1er vs 2e semestre)")
plt.legend()
plt.show()

# Comparaison entre l'âge des étudiants et le nombre de cours assistés en se basant sur les anomalies
col1 = "Age at enrollment"
col2 = "Course"

# Séparer les points normaux et anormaux
normal = df[predict == 1]
anomalies = df[predict == -1]

plt.figure(figsize=(7,5))

# Points normaux
plt.scatter(normal[col1], normal[col2], label="Normal")

# Points anormaux
plt.scatter(anomalies[col1], anomalies[col2], label="Anomalie")

plt.xlabel(col1)
plt.ylabel(col2)
plt.title("Isolation Forest sur l'âge en fonction du nombre de cours assistés")
plt.legend()
plt.show()