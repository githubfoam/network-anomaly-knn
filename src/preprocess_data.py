
---

### **src/preprocess_data.py**
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess(df, fit_scaler=False, scaler=None):
    df = df.copy()

    # Drop non-predictive columns if present
    drop_cols = ['id', 'attack_cat']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    # One-hot encode categorical features
    categorical_cols = ['proto', 'service', 'state']
    df = pd.get_dummies(df, columns=[col for col in categorical_cols if col in df.columns])

    # Separate features/label
    X = df.drop('label', axis=1)
    y = df['label']

    if fit_scaler:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    else:
        X = pd.DataFrame(scaler.transform(X), columns=X.columns)

    return X, y, scaler
