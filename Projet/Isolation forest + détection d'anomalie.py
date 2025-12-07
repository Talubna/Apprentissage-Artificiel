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