# Apprentissage-Artificiel

Warning : N'arrivant pas à faire fonctionner CookieCutter data science, je vais aussi déposer le projet en format .py en plus du format .ipynb.

Apprentissage artificiel d'un jeu de données traitant de l'assiduité des étudiants en utilisant plusieurs modèles d'apprentissage.

Pour cela, j'ai utilisé ce jeu de données qu'on peut retrouvé ici : https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

Il contient des données d'étudiants s'inscrivant à des univeersité au Portugal. Ils ont des données comme leur situation, leur moyens d'inscription, l'ordre de leur demande d'inscription, leur nationalité, leur rang dans leur promotion lors de leur inscription, leurs notes, les notes et leurs rangs des deux semestres, etc...
Ces données avaient pour but de pouvoir étudier et de pouvoir prédire celles et ceux qui vont être diplômés, inscrits ou celles et ceux qui vont abandonnés. La finalité étant de pouvoir offrir une aide à celle et ceux qui ont un risque d'échec.

Dans le cas de ce projet, j'ai utilisé ces données afin de pouvoir évaluer la véracité de ces données et l'entraîner sur plusieurs modèles différents (surtout supervisés car les données sont étiquettés). Notamment, les modèles ont l'arbre de décision (pour représenter clairement les données), la forêt aléatoire et l'isolation forest (avec la détection d'anomalies).
Sur un train / test de 80% / 20% (recommandé par ceux qui ont publié ces données), nous avons eu une précision plutôt satisfaisante (tous >7O% pour chaque modèle) avec pratiquement pas d'ajout d'hyperparamètres qui font baisser les mesures finales.

Les tests et les résultats sont disponibles sur le notebook Projet.ipynb ui est disponible dans le dossier Projet.
