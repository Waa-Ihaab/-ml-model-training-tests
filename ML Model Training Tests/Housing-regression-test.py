import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
import matplotlib.ticker as ticker

data = pd.read_csv('housing_sales_ma_ (1).csv')
data.head()

print(data.dtypes)

x = data['price_£']
y = data['surface']

# Filtrer les données pour la superficie entre 50 et 2500 m² et le prix entre 0 et 300,000 €
filtered_data = data[(data['surface'] >= 50) & (data['surface'] <= 2500) &
                     (data['price_£'] >= 30000) & (data['price_£'] <= 300000)]

# Définir les nouvelles variables filtrées
x_filtered = filtered_data['surface']
y_filtered = filtered_data['price_£']

# Visualiser uniquement les plages souhaitées
plt.figure(figsize=(10, 6))
plt.scatter(x_filtered, y_filtered, c='blue', alpha=0.5)
plt.xlabel('Superficie (m²)')
plt.ylabel('Prix (€)')
plt.title('Superficie&Prix des Logements au Maroc')

# Formatage des ticks pour afficher les prix
formatter = ticker.FuncFormatter(lambda x, _: f'{x:,.0f}')
plt.gca().yaxis.set_major_formatter(formatter)

# Limites des axes pour mieux cadrer la visualisation
plt.xlim([50, 2000])
plt.ylim([30000, 300000])

plt.grid(True)
plt.show()


############################################################################
############################################################################
############################################################################
############################################################################


# Normalisation des données
x = (x - np.mean(x)) / np.std(x)  # Normaliser x
y = (y - np.mean(y)) / np.std(y)  # Normaliser y


# Transformer x en une matrice 2D
X = x.values.reshape(-1, 1)  # Assurez-vous que x est un vecteur de forme (324, 1)

print(x.shape)
print(X.shape)
print(y.shape)


X = np.c_[np.ones(X.shape[0]), X]  # Ajouter une colonne de 1
print(X.shape)  # Doit afficher (324, 2)



#Theta
theta = np.array([[5], [0.01]])
theta


def model(X, theta):
    return X.dot(theta)


# Calculer les prédictions
y_pred = model(X, theta)



