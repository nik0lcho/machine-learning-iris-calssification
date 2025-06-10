from sklearn.preprocessing import StandardScaler, LabelEncoder


def preprocess_data(df):

    # Encode species (e.g. Iris Setosa --> 1)
    le = LabelEncoder()
    df['species'] = le.fit_transform(df['species'])

    # Scale features (e.g. [1, 2, 3] -> [-1.2, 0, 1.2])
    scaler = StandardScaler()
    X = scaler.fit_transform(df.drop('species', axis=1))
    y = df['species']

    # Return scaled features - X; Encoded labels - y; Label encoder; Scaler
    return X, y, le, scaler
