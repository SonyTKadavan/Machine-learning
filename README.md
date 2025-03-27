# ML Model Hub

A web application showcasing various machine learning models for different prediction tasks.

## Features

- Multiple Linear Regression: Employee Productivity Prediction
- Simple Linear Regression: Solar Panel Energy Output Prediction
- K-Nearest Neighbors: Health Insurance Premium Prediction
- Logistic Regression: Traffic Congestion Classification
- Polynomial Regression: Product Demand Prediction

## Local Development

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   python app.py
   ```
5. Open http://localhost:5000 in your browser

## Deployment on Render

1. Push your code to a Git repository (GitHub, GitLab, etc.)
2. Sign up for a Render account at https://render.com
3. Create a new Web Service
4. Connect your Git repository
5. Configure the service:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
   - Environment Variables:
     - PYTHON_VERSION: 3.9.0

## Project Structure

```
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── gunicorn_config.py    # Gunicorn configuration
├── render.yaml           # Render deployment config
├── templates/            # HTML templates
│   ├── index.html
│   ├── mlr.html
│   ├── slr.html
│   ├── knn.html
│   ├── logistic.html
│   └── poly.html
└── models/              # Trained ML models
    ├── mlr_model.pkl
    ├── slr_model.pkl
    ├── knn_model.pkl
    ├── logistic_model.pkl
    └── poly_model.pkl
```

## Technologies Used

- Python 3.9
- Flask
- NumPy
- Pandas
- Scikit-learn
- Gunicorn 