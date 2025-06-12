import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path

from src.data import save_data


def preprocess_data(df):

    # Encode species (e.g. Iris Setosa --> 1)
    le = LabelEncoder()
    df['species'] = le.fit_transform(df['species'])

    # Scale features (e.g. [1, 2, 3] -> [-1.2, 0, 1.2])
    scaler = StandardScaler()
    X = scaler.fit_transform(df.drop('species', axis=1))
    y = df['species']

    save_processed_data(X, y)

    # Return scaled features - X; Encoded labels - y; Label encoder; Scaler
    return X, y, le, scaler


def save_processed_data(X, y):

    project_root = Path(__file__).resolve().parents[1]

    # Define paths to save the processed files
    features_path = project_root / 'data' / 'processed' / 'processed_features.csv'
    labels_path = project_root / 'data' / 'processed' / 'processed_labels.csv'

    # Save processed features and labels
    columns_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    processed_features_df = pd.DataFrame(X, columns=columns_features)
    processed_labels_df = pd.DataFrame(y)
    save_data(processed_features_df, features_path)
    save_data(processed_labels_df, labels_path)

    return
