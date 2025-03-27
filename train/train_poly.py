import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('data/poly_product_demand.csv')

# Prepare features and target
X = df[['Social_Media_Mentions']]
y = df['Units_Sold']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create polynomial features
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Make predictions
y_pred = model.predict(X_test_poly)
accuracy = r2_score(y_test, y_pred)
print(f"Polynomial Regression Accuracy: {accuracy * 100:.2f}%")

# Save model, polynomial transformer, and scaler
model_data = {
    'model': model,
    'poly': poly,
    'scaler': scaler
}
pickle.dump(model_data, open("models/poly_model.pkl", "wb"))
