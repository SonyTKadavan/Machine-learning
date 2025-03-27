from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained models
mlr_model = pickle.load(open("models/mlr_model.pkl", "rb"))
slr_model = pickle.load(open("models/slr_model.pkl", "rb"))
knn_model = pickle.load(open("models/knn_model.pkl", "rb"))
logistic_model = pickle.load(open("models/logistic_model.pkl", "rb"))
poly_model = pickle.load(open("models/poly_model.pkl", "rb"))

# Home Page
@app.route('/')
def index():
    return render_template('index.html')

# MLR Route
@app.route('/mlr', methods=['GET', 'POST'])
def mlr():
    if request.method == 'POST':
        # Get base features
        features = {
            'Hours_Worked': float(request.form['hours']),
            'Virtual_Meetings': float(request.form['meetings']),
            'Break_Time': float(request.form['break']),
            'Internet_Speed': float(request.form['speed'])
        }
        
        # Create DataFrame and engineer features
        df = pd.DataFrame([features])
        df['Work_Intensity'] = df['Hours_Worked'] / (df['Break_Time'] + 1)
        df['Meeting_Load'] = df['Virtual_Meetings'] / (df['Hours_Worked'] + 1)
        df['Speed_Work_Ratio'] = df['Internet_Speed'] / (df['Hours_Worked'] + 1)
        df['Break_Work_Ratio'] = df['Break_Time'] / (df['Hours_Worked'] + 1)
        
        # Scale features
        df_scaled = mlr_model['scaler'].transform(df)
        df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
        
        # Select features
        df_selected = df_scaled[mlr_model['selected_features']]
        
        # Make prediction
        prediction = mlr_model['model'].predict(df_selected)[0]
        return render_template('mlr.html', result=round(prediction, 2))
    return render_template('mlr.html', result=None)

# SLR Route
@app.route('/slr', methods=['GET', 'POST'])
def slr():
    if request.method == 'POST':
        hours = float(request.form['hours'])
        prediction = slr_model.predict([[hours]])[0]
        return render_template('slr.html', result=round(prediction, 2))
    return render_template('slr.html', result=None)

# KNN Route
@app.route('/knn', methods=['GET', 'POST'])
def knn():
    if request.method == 'POST':
        # Get features
        features = {
            'BMI': float(request.form['bmi']),
            'Age': float(request.form['age']),
            'Smoking_Status': float(request.form['smoking']),
            'Exercise_Frequency': float(request.form['exercise'])
        }
        
        # Create DataFrame
        df = pd.DataFrame([features])
        
        # Scale features
        df_scaled = knn_model['scaler'].transform(df)
        
        # Make prediction
        prediction = knn_model['model'].predict(df_scaled)[0]
        return render_template('knn.html', result=round(prediction, 2))
    return render_template('knn.html', result=None)

# Logistic Regression Route
@app.route('/logistic', methods=['GET', 'POST'])
def logistic():
    if request.method == 'POST':
        # Get features
        features = {
            'Vehicle_Count': float(request.form['vehicles'])
        }
        
        # Create DataFrame
        df = pd.DataFrame([features])
        
        # Scale features
        df_scaled = logistic_model['scaler'].transform(df)
        
        # Make prediction
        prediction = logistic_model['model'].predict(df_scaled)[0]
        return render_template('logistic.html', result=prediction)
    return render_template('logistic.html', result=None)

# Polynomial Regression Route
@app.route('/poly', methods=['GET', 'POST'])
def poly():
    if request.method == 'POST':
        mentions = float(request.form['mentions'])
        
        # Create DataFrame
        df = pd.DataFrame([[mentions]], columns=['Social_Media_Mentions'])
        
        # Scale features
        df_scaled = poly_model['scaler'].transform(df)
        
        # Create polynomial features
        df_poly = poly_model['poly'].transform(df_scaled)
        
        # Make prediction
        prediction = poly_model['model'].predict(df_poly)[0]
        return render_template('poly.html', result=round(prediction, 2))
    return render_template('poly.html', result=None)

if __name__ == '__main__':
    app.run(debug=True) 