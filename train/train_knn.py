import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('data/knn_health_insurance.csv')

# Prepare features and target
X = df[['BMI', 'Age', 'Smoking_Status', 'Exercise_Frequency']]
y = df['Insurance_Premium']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)
print(f"KNN Model Accuracy: {accuracy * 100:.2f}%")

# Save model and scaler
model_data = {
    'model': model,
    'scaler': scaler
}
pickle.dump(model_data, open("models/knn_model.pkl", "wb"))
