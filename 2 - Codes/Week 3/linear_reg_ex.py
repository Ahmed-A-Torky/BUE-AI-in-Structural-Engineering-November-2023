# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Creating a dataset
# Assume we have a dataset of beams where we know their width, height, load and stress
data = {
    'width': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    'height': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'load': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    'stress': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]  # Hypothetical stress values
}
df = pd.DataFrame(data)

# Defining features (width, height, load) and target (stress)
X = df[['width', 'height', 'load']]
y = df['stress']

# Splitting the dataset into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the linear regression model and training it
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions on test data
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Predicting the stress in a new beam with specific dimensions and load
new_beam = np.array([[60, 120, 1200]])    # Width: 60, Height: 120, Load: 1200
predicted_stress = model.predict(new_beam)
print(f'Predicted Stress for the new beam: {predicted_stress[0]}')