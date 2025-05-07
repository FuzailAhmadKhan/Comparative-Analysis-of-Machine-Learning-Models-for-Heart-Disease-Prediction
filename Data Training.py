import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Load preprocessed data
data = np.load('preprocessed_data.npz')
X_train = data['X_train']
y_train = data['y_train']

# Initialize models with optimized parameters
models = {
    'Decision Tree': DecisionTreeClassifier(
        max_depth=3,
        random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    ),
    'Logistic Regression': LogisticRegression(
        max_iter=1000,  # Prevents convergence warning
        random_state=42
    ),
    'SVM': SVC(
        kernel='rbf',  # Radial Basis Function kernel
        probability=True,  # Required for ROC curves
        random_state=42
    ),
    'KNN': KNeighborsClassifier(
        n_neighbors=5  # Default neighbors
    )
}

# Train and save models
for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f'{name.lower().replace(" ", "_")}_model.pkl')
    print(f"✅ {name} trained and saved.")

print("\n🔥 All models trained successfully!")