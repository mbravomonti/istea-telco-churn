import pandas as pd
import joblib
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train import preprocess_for_training

# Define paths
PROCESSED_DATA_PATH = os.path.join('data', 'processed', 'telco_churn_clean.csv')
MODEL_PATH = os.path.join('models', 'model.pkl')
METRICS_PATH = 'metrics.json'
EVAL_PLOTS_DIR = 'eval_plots'

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        raise FileNotFoundError(f"{MODEL_PATH} not found. Run train.py first.")

def load_data():
    if os.path.exists(PROCESSED_DATA_PATH):
        return pd.read_csv(PROCESSED_DATA_PATH)
    else:
        raise FileNotFoundError(f"{PROCESSED_DATA_PATH} not found.")

def evaluate_model(model, df):
    # Preprocess data (ensure consistency with training)
    df_processed = preprocess_for_training(df)
    
    X = df_processed.drop(columns=['churn'])
    y = df_processed['churn']
    
    # Predict probabilities
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(EVAL_PLOTS_DIR, 'roc_curve.png'))
    plt.close()
    
    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(EVAL_PLOTS_DIR, 'confusion_matrix.png'))
    plt.close()
    
    print(f"Evaluation plots saved to {EVAL_PLOTS_DIR}")

if __name__ == "__main__":
    try:
        os.makedirs(EVAL_PLOTS_DIR, exist_ok=True)
        
        print("Loading model...")
        model = load_model()
        
        print("Loading data...")
        df = load_data()
        
        print("Evaluating model...")
        evaluate_model(model, df)
        
    except Exception as e:
        print(f"Error: {e}")
