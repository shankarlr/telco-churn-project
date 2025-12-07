# inspect_model.py
import joblib
import pandas as pd

model = joblib.load("models/simple_model.joblib")
preproc = model.named_steps['preproc']
clf = model.named_steps['clf']

# numeric features
numeric = preproc.transformers_[0][2]  # list of numeric column names
# After onehot, coefficients are in a vector; we show numeric first
num_coefs = clf.coef_[0][:len(numeric)]
print("Numeric coefficients:")
for name, coef in zip(numeric, num_coefs):
    print(f"  {name}: {coef:.4f}   -> {'increases' if coef>0 else 'decreases'} churn")

# Show categorical mapping (contract)
if len(preproc.transformers_) > 1:
    # get one-hot feature names
    ohe = preproc.named_transformers_['cat']
    try:
        cat_names = ohe.get_feature_names_out(preproc.transformers_[1][2])
        cat_coefs = clf.coef_[0][len(numeric):]
        print("\nCategorical coefficients (Contract):")
        for name, coef in zip(cat_names, cat_coefs):
            print(f"  {name}: {coef:.4f}   -> {'increases' if coef>0 else 'decreases'} churn")
    except Exception:
        print("Could not extract categorical names. The model may use a different transformer API.")
