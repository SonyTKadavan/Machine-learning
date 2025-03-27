import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load data
print("Loading data...")
df = pd.read_csv('data/slr_solar_panel_efficiency.csv')

X = df[['Sunlight_Hours']]
y = df['Energy_Output_kWh']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
print("\nTraining model...")
model = LinearRegression()
model.fit(X_train, y_train)

# Cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
print("\nCross-validation scores:", cv_scores)
print(f"Mean CV score: {np.mean(cv_scores) * 100:.2f}%")

# Test set prediction
y_pred = model.predict(X_test)
test_accuracy = r2_score(y_test, y_pred)
print(f"Test set accuracy: {test_accuracy * 100:.2f}%")

# Print model parameters
print(f"\nModel coefficients:")
print(f"Slope (m): {model.coef_[0]:.2f}")
print(f"Intercept (b): {model.intercept_:.2f}")

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('Sunlight Hours')
plt.ylabel('Energy Output (kWh)')
plt.title('Sunlight Hours vs Energy Output: Linear Regression')
plt.legend()
plt.savefig('models/slr_visualization.png')
plt.close()

# Save model
print("\nSaving model...")
pickle.dump(model, open("models/slr_model.pkl", "wb"))
print("Done!")
