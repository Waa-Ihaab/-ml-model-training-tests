#import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
import matplotlib.ticker as ticker

#read file
data = pd.read_csv('housing_sales_ma_ (1).csv')

#print first 5 ellements
data.head()

#print types of the columns
print(data.dtypes)

#define x and y
x = data['price_£']
y = data['surface']

# Filter the dataset to remove outliers and extreme values for better visualization
filtered_data = data[(data['surface'] >= 50) & (data['surface'] <= 2500) &
                     (data['price_£'] >= 30000) & (data['price_£'] <= 300000)]

# Redefine x and y based on the filtered data
x_filtered = filtered_data['surface']
y_filtered = filtered_data['price_£']

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(x_filtered, y_filtered, c='blue', alpha=0.5)
plt.xlabel('Superficie (m²)')
plt.ylabel('Prix (€)')
plt.title('Relation entre la Superficie et le Prix des Logements au Maroc')

# Format the Y-axis to display numbers with commas (e.g., 150,000)
formatter = ticker.FuncFormatter(lambda x, _: f'{x:,.0f}')
plt.gca().yaxis.set_major_formatter(formatter)

# Set axis limits
plt.xlim([50, 2500])
plt.ylim([0, 300000])

plt.grid(True)
plt.show()
