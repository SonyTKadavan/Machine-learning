import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('data/logistic_traffic_congestion.csv')

# Prepare features and target
X = df[['Vehicle_Count']]
y = df['Congestion_Level']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Accuracy: {accuracy * 100:.2f}%")

# Save model and scaler
model_data = {
    'model': model,
    'scaler': scaler
}
pickle.dump(model_data, open("models/logistic_model.pkl", "wb"))
