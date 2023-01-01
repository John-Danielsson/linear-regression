# Import libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from statsmodels.formula.api import ols
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Load data
# Make sure 'World_Energy_Consumption.csv' is in the directory
df = pd.read_csv('World_Energy_Consumption.csv')
columns = [
    'population',
    'gdp',
    'energy_per_capita',
    'energy_per_gdp'
]

# Feature Engineering / Data Wrangling
df = df[columns].dropna()
df['gdp_per_capita'] = df['gdp'] / df['population']
# Remove outliers
# Top 5% GDP
g_t = df['gdp_per_capita'].quantile(0.95)
# Bottom 5% GDP
g_b = df['gdp_per_capita'].quantile(0.05)
# Top 5% energy
e_t = df['energy_per_capita'].quantile(0.95)
# Bottom 5% energy
e_b = df['energy_per_capita'].quantile(0.05)
df = df[(g_b < df['gdp_per_capita']) & (df['gdp_per_capita'] < g_t)]
df = df[(e_b < df['energy_per_capita']) & (df['energy_per_capita'] < e_t)]

# Average GDP per capita divided by average energy use per capita
baseline = np.average(df['gdp_per_capita'])
baseline /= np.average(df['energy_per_capita'])
df['baseline'] = range(df.shape[0]) * baseline

# Make independent (X) and dependent (y) variables
X = np.array(df['energy_per_capita'])[:, np.newaxis]
y = df['gdp_per_capita']

# Split data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)
# print(X_train.shape)

# Set up Linear Regression
model = LinearRegression()
model.fit(X=X_train, y=y_train)
slope = model.coef_
intercept = model.intercept_
print(f'Predicted GDP/capita = (Energy use per capita)*{slope} + {intercept}')

# Predict GDP per capita
y_pred = model.predict(X_test)
print(f'      LR r^2 = {r2_score(y_test, y_pred)}')
baseline_predict = [baseline*x for x in y_pred]
print(f'baseline r^2 = {r2_score(baseline_predict, y_pred)}')
print(f'        rmse = {sqrt(mse(y_test, y_pred))}')

# Data Visualization
sns.regplot(x=X_train, y=y_train, line_kws={"color": "red"})
energy_range = range(int(max(df['energy_per_capita'])))
sns.lineplot(
    x=energy_range,
    y=[int(baseline*i) for i in energy_range],
    color='black'
)
plt.xlabel('Energy use per capita')
plt.ylabel('GDP per capita')
black_patch = Patch(color='black', label='baseline')
red_patch = Patch(color='red', label='regression')
plt.legend(handles=[black_patch, red_patch], loc=(0.02, 0.8))
plt.show()
