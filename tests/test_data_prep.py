from src.data_prep import clean
import pandas as pd

def test_totalcharges_handling():
    df = pd.DataFrame({
        "customerID": ["1"],
        "tenure": [1],
        "MonthlyCharges": [20.0],
        "TotalCharges": [" "],
        "Churn": ["No"]
    })
    out = clean(df)
    assert out['TotalCharges'].iloc[0] == 0
