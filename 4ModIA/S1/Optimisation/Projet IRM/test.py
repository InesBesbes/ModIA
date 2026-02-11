import numpy as np

# Fonction pour calculer le gradient du risque empirique pour un seul échantillon
def grad_En_sample(w, x, y):
    return (np.dot(w, x) - y) * x

# Algorithme de gradient stochastique à pas constant
def stochastic_gradient_descent(X, Y, learning_rate, n_epochs):
    d = X.shape[1]  # Dimension des données
    n = X.shape[0]  # Nombre d'échantillons
    w = np.zeros(d)  # Initialisation du vecteur de poids

    En_wk = []  # Pour stocker les valeurs du risque empirique En(wk)
    E_wk = []   # Pour stocker les valeurs du risque moyen E(wk)

    for epoch in range(n_epochs):
        for i in range(n):
            # Sélection aléatoire d'un échantillon
            random_index = np.random.randint(n)
            x = X[random_index]
            y = Y[random_index]

            # Calcul du gradient sur un seul échantillon
            gradient = grad_En_sample(w, x, y)

            # Mise à jour des poids
            w = w - learning_rate * gradient

        # Calcul et stockage du risque empirique et du risque moyen à la fin de chaque epoch
        En_wk.append(np.mean((np.dot(X, w) - Y) ** 2) / 2)
        E_wk.append(np.mean((np.dot(X, w) - Y) ** 2) / 2)

    return w, En_wk, E_wk

import numpy as np

# Fonction pour calculer le gradient du risque empirique pour un seul échantillon
def grad_En_sample(w, x, y):
    return (np.dot(w, x) - y) * x

# Algorithme de gradient stochastique à pas décroissant
def stochastic_gradient_descent_decay(X, Y, initial_learning_rate, n_epochs):
    d = X.shape[1]  # Dimension des données
    n = X.shape[0]  # Nombre d'échantillons
    w = np.zeros(d)  # Initialisation du vecteur de poids
    learning_rate = initial_learning_rate

    En_wk = []  # Pour stocker les valeurs du risque empirique En(wk)
    E_wk = []   # Pour stocker les valeurs du risque moyen E(wk)

    for epoch in range(n_epochs):
        for i in range(n):
            # Sélection aléatoire d'un échantillon
            random_index = np.random.randint(n)
            x = X[random_index]
            y = Y[random_index]

            # Calcul du gradient sur un seul échantillon
            gradient = grad_En_sample(w, x, y)

            # Mise à jour des poids avec pas décroissant
            w = w - learning_rate * gradient

            # Mise à jour du taux d'apprentissage à chaque itération
            learning_rate = initial_learning_rate / (1 + epoch * n + i)

        # Calcul et stockage du risque empirique et du risque moyen à la fin de chaque epoch
        En_wk.append(np.mean((np.dot(X, w) - Y) ** 2) / 2)
        E_wk.append(np.mean((np.dot(X, w) - Y) ** 2) / 2)

    return w, En_wk, E_wk

import numpy as np
import matplotlib.pyplot as plt

# Générer des données pour l'expérience
# ...

# Effectuer l'algorithme de gradient stochastique à pas constant sur En
# ...

# Effectuer l'algorithme de gradient stochastique à pas décroissant sur En
# ...

# Stocker les suites En(wk) et E(wk) pour chaque méthode
# ...

# Tracer les courbes de convergence
epochs = range(n_epochs)  # Remplacer n_epochs par le nombre réel d'époques
plt.plot(epochs, En_wk_constant, label='Pas constant')
plt.plot(epochs, En_wk_decay, label='Pas décroissant')
plt.xlabel('Epochs')
plt.ylabel('Valeur du risque empirique')
plt.title('Convergence du risque empirique en fonction du nombre d\'epochs')
plt.legend()
plt.show()

import numpy as np

# Algorithme de gradient stochastique en ligne
def online_stochastic_gradient_descent(X, Y, learning_rate, n_epochs):
    d = X.shape[1]  # Dimension des données
    n = X.shape[0]  # Nombre d'échantillons
    w = np.zeros(d)  # Initialisation du vecteur de poids

    En_wk = []  # Pour stocker les valeurs du risque empirique En(wk)
    E_wk = []   # Pour stocker les valeurs du risque moyen E(wk)

    for epoch in range(n_epochs):
        for i in range(n):
            # Sélection aléatoire d'un échantillon
            random_index = np.random.randint(n)
            x = X[random_index]
            y = Y[random_index]

            # Calcul du gradient sur un seul échantillon
            gradient = (np.dot(w, x) - y) * x

            # Mise à jour des poids
            w = w - learning_rate / np.sqrt(epoch * n + i + 1) * gradient

        # Calcul et stockage du risque empirique et du risque moyen à la fin de chaque epoch
        En_wk.append(np.mean((np.dot(X, w) - Y) ** 2) / 2)
        E_wk.append(np.mean((np.dot(X, w) - Y) ** 2) / 2)

    return w, En_wk, E_wk

import numpy as np

# Algorithme SAGA
def saga(X, Y, learning_rate, n_epochs):
    d = X.shape[1]  # Dimension des données
    n = X.shape[0]  # Nombre d'échantillons
    w = np.zeros(d)  # Initialisation du vecteur de poids
    gradient_memory = np.zeros((n, d))  # Pour stocker les gradients passés
    a = np.zeros(d)  # Pour stocker les moyennes des gradients passés

    En_wk = []  # Pour stocker les valeurs du risque empirique En(wk)
    E_wk = []   # Pour stocker les valeurs du risque moyen E(wk)

    for epoch in range(n_epochs):
        for i in range(n):
            # Sélection aléatoire d'un échantillon
            random_index = np.random.randint(n)
            x = X[random_index]
            y = Y[random_index]

            # Calcul du gradient sur un seul échantillon
            gradient = (np.dot(w, x) - y) * x

            # Mise à jour des moyennes des gradients passés
            a += (gradient - gradient_memory[random_index]) / n

            # Mise à jour des poids
            w = w - learning_rate * a

            # Mise à jour du gradient dans la mémoire
            gradient_memory[random_index] = gradient

        # Calcul et stockage du risque empirique et du risque moyen à la fin de chaque epoch
        En_wk.append(np.mean((np.dot(X, w) - Y) ** 2) / 2)
        E_wk.append(np.mean((np.dot(X, w) - Y) ** 2) / 2)

    return w, En_wk, E_wk