# src/model.py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import joblib

DATA_PATH = Path("data/processed_with_features.csv")
MODEL_PATH = Path("models/simple_model.joblib")

def load_data(path: Path = DATA_PATH):
    if not path.exists():
        raise FileNotFoundError("Processed data not found. Run features first.")
    return pd.read_csv(path)

def prepare_xy(df: pd.DataFrame):
    df = df.copy()
    # convert Churn to 0/1 if needed
    if df['Churn'].dtype == object:
        df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})
    # features we will use (simple, explainable)
    features = [f for f in ['tenure','MonthlyCharges','TotalCharges','num_services','Contract'] if f in df.columns]
    X = df[features]
    y = df['Churn']
    return X, y

def train_and_save(model_out: Path = MODEL_PATH):
    df = load_data()
    X, y = prepare_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=16)
    numeric = [c for c in X.select_dtypes(include=['int64','float64']).columns]
    categorical = [c for c in X.select_dtypes(include=['object']).columns]
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical)
    ])
    pipe = Pipeline([("preproc", preprocessor), ("clf", LogisticRegression(max_iter=500))])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:,1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "auc": float(roc_auc_score(y_test, probs)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist()
    }
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_out)
    print(f"Saved model to {model_out}")
    print("Metrics:", metrics)
    return metrics

if __name__ == "__main__":
    train_and_save()
