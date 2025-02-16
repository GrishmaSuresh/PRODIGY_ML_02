import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Load dataset from a CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    df = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    return scaled_data

if __name__ == "__main__":
    df = load_data("../data/Mall_Customers.csv")
    scaled_data = preprocess_data(df)
    print("Data Preprocessed Successfully!")
