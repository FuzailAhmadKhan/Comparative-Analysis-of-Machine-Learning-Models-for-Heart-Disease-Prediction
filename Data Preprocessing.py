import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load and clean data
df = pd.read_csv('heart.csv')

# Handle missing values (if any exist)
df = df.replace('?', np.nan)
df = df.dropna()

# Convert all columns to numeric (some might be read as objects)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='ignore')

# Split data
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save processed data
np.savez('preprocessed_data.npz', 
         X_train=X_train, X_test=X_test,
         y_train=y_train, y_test=y_test)
joblib.dump(scaler, 'scaler.pkl')

print("Preprocessing completed successfully!")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")