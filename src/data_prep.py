import pandas as pd
import os

# Define paths
RAW_DATA_PATH = os.path.join('data', 'raw', 'telco_churn.csv')
SOURCE_DATA_PATH = 'telco_churn.csv'

def load_data():
    """
    Load data from the source CSV file.
    If the file exists in the root, it reads it.
    Ideally, in a DVC pipeline, this would read from data/raw.
    """
    if os.path.exists(SOURCE_DATA_PATH):
        print(f"Loading data from {SOURCE_DATA_PATH}...")
        df = pd.read_csv(SOURCE_DATA_PATH)
        return df
    elif os.path.exists(RAW_DATA_PATH):
        print(f"Loading data from {RAW_DATA_PATH}...")
        df = pd.read_csv(RAW_DATA_PATH)
        return df
    else:
        raise FileNotFoundError("Dataset not found in root or data/raw/")

def clean_data(df):
    """
    Basic data cleaning and preprocessing.
    """
    # Copy dataframe to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Convert TotalCharges to numeric (it might be object type due to empty strings)
    df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')
    
    # Fill missing values for TotalCharges with 0 or median
    df['total_charges'] = df['total_charges'].fillna(0)
    
    # Normalize text columns if necessary
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    return df

if __name__ == "__main__":
    try:
        df = load_data()
        print(f"Data loaded successfully. Shape: {df.shape}")
        
        df_clean = clean_data(df)
        print("Data cleaning completed.")
        print(df_clean.head())
        
        # Save cleaned data
        output_path = os.path.join('data', 'processed', 'telco_churn_clean.csv')
        df_clean.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
