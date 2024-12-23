import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
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
plt.title('Relation entre la Superficie et le Prix des Logements au Maroc')

# Formatage des ticks pour afficher les prix
formatter = ticker.FuncFormatter(lambda x, _: f'{x:,.0f}')
plt.gca().yaxis.set_major_formatter(formatter)

# Limites des axes pour mieux cadrer la visualisation
plt.xlim([50, 2500])
plt.ylim([0, 300000])

plt.grid(True)
plt.show()
