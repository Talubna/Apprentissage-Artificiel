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

# Bibliothèques pour la forêt aléatoire
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Normalisation des variables
df['Target'] = df['Target'].map({'Dropout':0, 'Enrolled':1, 'Graduated':2})

# Classification par forêt aléatoire
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("Évaluation en moyenne :")
print("Exactitude :", accuracy_score(y_test, y_pred))
print("Précision :", precision_score(y_test, y_pred, average='weighted'))
print("Rappel :", recall_score(y_test, y_pred, average='weighted'))
print("F1 score :", f1_score(y_test, y_pred, average='weighted'))
print(classification_report(y_test, y_pred))

# Hyperparamètres
param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,20)}

# Create a random model classifier
rf = RandomForestClassifier()

# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(rf, 
                                 param_distributions = param_dist, 
                                 n_iter=5, 
                                 cv=5)

# Fit the random search object to the model
rand_search.fit(X_train, y_train)

# Create a variable for the best model
best_rf = rand_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  rand_search.best_params_)

# Create a series containing feature importances from the model and feature names from the training model
# Convert X_train to DataFrame if it's a numpy array, or use column names from elsewhere
if isinstance(X_train, np.ndarray):
    # Generate feature names that match the number of features in your model
    feature_names = [f'feature_{i}' for i in range(len(best_rf.feature_importances_))]
    
    feature_importances = pd.Series(best_rf.feature_importances_, index=feature_names).sort_values(ascending=False)
else:
    # If X_train is already a DataFrame, use its columns
    feature_importances = pd.Series(best_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# Plot a simple bar chart
feature_importances.plot.bar();

# Generate predictions with the best model
y_pred = best_rf.predict(X_test)

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)

ConfusionMatrixDisplay(confusion_matrix=cm).plot();