import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
# 1. Load the dataset
df = pd.read_csv('advertising.csv')
df.head()
# 2. Explore the dataset
df.columns.tolist()
df.info()
print(df.info())
print(df.describe())
df.isnull().sum()
df.fillna(0, inplace=True)
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation matrix')
plt.show()
# 3. Visualize the relationships between features and target variable
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 4. Train a linear regression model
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
# 5. Evaluate the model
from sklearn.metrics import mean_squared_error, r2_score
y_pred = reg_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
