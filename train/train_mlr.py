import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')

print("Loading and preparing data...")
# Load dataset
df = pd.read_csv('data/mlr_employee_productivity.csv')

# Define features and target
X = df[['Hours_Worked', 'Virtual_Meetings', 'Break_Time', 'Internet_Speed']]
y = df['Productivity_Score']

# Feature engineering focused on business logic
print("Performing feature engineering...")
X['Work_Intensity'] = X['Hours_Worked'] / (X['Break_Time'] + 1)
X['Meeting_Load'] = X['Virtual_Meetings'] / (X['Hours_Worked'] + 1)
X['Speed_Work_Ratio'] = X['Internet_Speed'] / (X['Hours_Worked'] + 1)
X['Break_Work_Ratio'] = X['Break_Time'] / (X['Hours_Worked'] + 1)

# Scale features
print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split dataset
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, shuffle=True
)

# Create and train initial model for feature selection
print("Performing feature selection...")
selector = RandomForestRegressor(n_estimators=100, random_state=42)
selector.fit(X_train, y_train)

# Select most important features
selector = SelectFromModel(selector, prefit=True, threshold='mean')
feature_idx = selector.get_support()
selected_features = X.columns[feature_idx].tolist()
print(f"\nSelected features: {selected_features}")

# Train final model with selected features
print("\nTraining final model...")
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

model.fit(X_train_selected, y_train)

# Cross-validation
print("\nPerforming cross-validation...")
cv_scores = cross_val_score(model, X_train_selected, y_train, cv=5, scoring='r2')
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {np.mean(cv_scores) * 100:.2f}%")

# Test set prediction
print("\nEvaluating on test set...")
y_pred = model.predict(X_test_selected)
test_accuracy = r2_score(y_test, y_pred)
print(f"Test set accuracy: {test_accuracy * 100:.2f}%")

# Feature importance for selected features
feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': model.feature_importances_
})
print("\nFeature Importance:")
print(feature_importance.sort_values('importance', ascending=False))

# Save model and feature selector
print("\nSaving model and feature selector...")
model_data = {
    'model': model,
    'selector': selector,
    'scaler': scaler,
    'selected_features': selected_features
}
pickle.dump(model_data, open("models/mlr_model.pkl", "wb"))
print("Done!")
