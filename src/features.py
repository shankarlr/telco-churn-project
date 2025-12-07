# src/features.py
import pandas as pd
from pathlib import Path

PROCESSED_IN = Path("data/processed_telco.csv")
PROCESSED_OUT = Path("data/processed_with_features.csv")

def load_processed(path: Path = PROCESSED_IN):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run data_prep first.")
    return pd.read_csv(path)

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # tenure_group: 0-6, 7-24, 25+
    df['tenure_group'] = pd.cut(df['tenure'], bins=[-1,6,24,72], labels=['0-6','7-24','25+'])
    # service columns - make sure they are numeric 1/0 (after cleaning)
    service_cols = ['PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
                    'DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    present_services = [c for c in service_cols if c in df.columns]
    # If already numeric, keep; else map strings starting with 'Y' to 1
    for c in present_services:
        df[c] = df[c].apply(lambda x: 1 if str(x).lower().startswith('y') or x==1 else 0)
    df['num_services'] = df[present_services].sum(axis=1)
    return df

def save(df: pd.DataFrame, path: Path = PROCESSED_OUT):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved features file to {path}")

def run():
    df = load_processed()
    df = add_features(df)
    save(df)

if __name__ == "__main__":
    run()
