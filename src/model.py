from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


def train_model(X_train, y_train):
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Return accuracy score."""
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)


def save_model(model, path):
    """Save model to disk."""
    joblib.dump(model, path)

# Example usage:
# model = train_model(X_train, y_train)
# save_model(model, "model.pkl")
