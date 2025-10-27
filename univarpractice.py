import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load and clean data
df = pd.read_csv('/Users/shraddhap/Desktop/diabetic_data.csv')
df['readmission_binary'] = df['readmitted'].replace({'NO': 0, '>30': 0, '<30': 1})
df.replace('?', pd.NA, inplace=True)
df = df.dropna()

# Drop irrelevant columns
df = df.drop(columns=[
    'encounter_id', 'patient_nbr', 'weight', 'admission_type_id',
    'discharge_disposition_id', 'admission_source_id', 'payer_code',
    'medical_specialty', 'diag_1', 'diag_2', 'diag_3',
    'max_glu_serum', 'A1Cresult'
])

# Define target
y = df['readmission_binary']

# Loop through each feature
for col in df.columns:
    if col in ['readmitted', 'readmission_binary']:
        continue

    X = df[[col]]

    # One-hot encode if categorical
    if X[col].dtype == 'object':
        encoder = OneHotEncoder(drop='first', sparse=False)
        X_encoded = encoder.fit_transform(X)
        feature_names = encoder.get_feature_names_out([col])
    else:
        X_encoded = X.values
        feature_names = [col]

    # Skip if encoding results in zero features
    if X_encoded.shape[1] == 0:
        print(f"Skipping {col}: no usable features after encoding.")
        continue

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # Fit logistic regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Print odds ratios
    odds_ratios = np.exp(model.coef_[0])
    print(f"\nFeature: {col}")
    for name, or_val in zip(feature_names, odds_ratios):
        print(f"  {name}: Odds Ratio = {or_val:.2f}")
