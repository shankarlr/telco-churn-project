import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/telco_churn.csv")
PROCESSED_PATH = Path("data/processed_telco.csv")

def load_raw(path: Path = RAW_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Raw file not found at {path}. Please download dataset and place it there.")
    return pd.read_csv(path)

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.select_dtypes(include=['object']).columns:
        df[c] = df[c].astype(str).str.strip()
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', ''), errors='coerce').fillna(0)
    for c in df.columns:
        sample = df[c].dropna().astype(str).unique()[:5].tolist()
        if set(sample) & {'Yes','No'}:
            df[c] = df[c].map({'Yes':1, 'No':0})
    return df

def save(df: pd.DataFrame, path: Path = PROCESSED_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved processed file to {path}")

def run():
    df = load_raw()
    df = clean(df)
    save(df)

if __name__ == "__main__":
    run()
