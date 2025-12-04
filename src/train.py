import pandas as pd
import yaml
import json
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Define paths
PROCESSED_DATA_PATH = os.path.join('data', 'processed', 'telco_churn_clean.csv')
METRICS_PATH = 'metrics.json'
MODEL_PATH = os.path.join('models', 'model.pkl')
PARAMS_PATH = 'params.yaml'

def load_params():
    with open(PARAMS_PATH, 'r') as f:
        params = yaml.safe_load(f)
    return params

def load_data():
    if os.path.exists(PROCESSED_DATA_PATH):
        return pd.read_csv(PROCESSED_DATA_PATH)
    else:
        raise FileNotFoundError(f"{PROCESSED_DATA_PATH} not found. Run data_prep.py first.")

def preprocess_for_training(df):
    # Simple preprocessing for the baseline model
    # Drop customer_id as it's not a feature
    df = df.drop(columns=['customer_id'], errors='ignore')
    
    # Encode categorical variables
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
        
    return df

def train_model(df, params):
    X = df.drop(columns=['churn'])
    y = df['churn']
    
    test_size = params['train']['test_size']
    random_state = params['train']['random_state']
    model_params = params['train']['model_params']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    model = LogisticRegression(**model_params)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    return model, metrics

if __name__ == "__main__":
    try:
        print("Loading parameters...")
        params = load_params()
        
        print("Loading data...")
        df = load_data()
        
        print("Preprocessing data...")
        df_processed = preprocess_for_training(df)
        
        print("Training model...")
        model, metrics = train_model(df_processed, params)
        
        print("Saving metrics...")
        with open(METRICS_PATH, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(metrics)
        
        print("Saving model...")
        joblib.dump(model, MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
        
    except Exception as e:
        print(f"Error: {e}")
