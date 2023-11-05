# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Creating a dataset
# Assume we have a dataset of steel columns where we know their length, cross-sectional area and axial capacity
data = {
    'length': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],  # in meters
    'area': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # in square meters
    'capacity': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]  # in kN
}
df = pd.DataFrame(data)

# Defining features (length, area) and target (capacity)
X = df[['length', 'area']].values
y = df['capacity'].values

# Splitting the dataset into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, 
                                                    random_state=42)

# Creating the linear regression model and training it
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions on the test data
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Predicting the axial capacity of a new steel column with specific length and cross-sectional area
new_column = np.array([[22, 1.1]])  # Length: 22m, Area: 1.1m^2
predicted_capacity = model.predict(new_column)
print(f'Predicted Axial Capacity for the new column: {predicted_capacity[0]} kN')