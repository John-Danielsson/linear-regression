# Import libraries
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_absolute_error as mae
from statsmodels.formula.api import ols
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Load data
# Make sure 'World_Energy_Consumption.csv' is in the directory
df = pd.read_csv('World_Energy_Consumption.csv')
columns = [
    'country',
    'year',
    'population',
    'gdp',
    'energy_per_capita',
    'energy_per_gdp'
]

# Feature Engineering / Data Wrangling
df = df[columns].dropna()
df['gdp_per_capita'] = df['gdp'] / df['population']
# Average GDP per capita divided by average energy use per capita
baseline = np.average(df['gdp_per_capita'])
baseline /= np.average(df['energy_per_capita'])
df['baseline'] = range(df.shape[0]) * baseline

# Linear Regression
# Dependent variable
y = df['gdp_per_capita']
# Independent variable
x = df['energy_per_capita']
# Linear regression model (Ordinary Least Squares)
lr = ols('y ~ x', data=df).fit()
params = lr.params
intercept = params[0]
slope = params[1]
print('LINE OF BEST FIT')
print(f'            baseline slope = {baseline}')
print(f'    line-of-best-fit slope = {slope}')
print(f'line-of-best-fit intercept = {intercept}')
print()

# Compare predictive power of baseline and line-of-best-fit prediction
r2_x_y = pearsonr(x=x, y=y)
r2_baseline_y = pearsonr(x=x, y=df['baseline'])
print('CORRELATION')
print('Correlation for GDP per capita (y) and energy per capita (x)')
print(f'Pearson\'s correlation coefficient = {r2_x_y[0]}')
print('Correlation for baseline (y) and energy per capita (x)')
print(f'Pearson\'s correlation coefficient = {r2_baseline_y[0]}')
print()

# Add y_pred column to df
df['y_pred'] = df['energy_per_capita'] * slope + intercept

# Calculate evaluation metrics for baseline and line-of-best-fit
rmse_baseline = sqrt(mae(df['baseline'], df['gdp_per_capita']))
rmse_lobf = sqrt(mae(df['y_pred'], df['gdp_per_capita']))
print('EVALUATION METRICS')
print(f'        RMSE for baseline = {rmse_baseline}')
print(f'RMSE for line-of-best-fit = {rmse_lobf}')
print()

# Data visualization
grid = sns.JointGrid(x=x, y=y, space=0, height=6, ratio=50)
grid.plot_joint(plt.scatter, color="g")
plt.plot(
    [0, 3e5],
    [0, 3e5*baseline],
    linewidth=2,
    label='baseline',
    color='blue'
)
plt.plot(
    [0, 3e5],
    [intercept, intercept+3e5*slope],
    linewidth=2,
    label='regression',
    color='red'
)
blue_patch = Patch(color='blue', label='baseline')
red_patch = Patch(color='red', label='regression')
plt.legend(handles=[blue_patch, red_patch], loc=(0.02, 0.6))
plt.show()
