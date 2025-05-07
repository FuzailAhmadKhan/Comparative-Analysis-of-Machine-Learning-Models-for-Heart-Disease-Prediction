#  Heart Disease Prediction - ML Model Comparison

This project focuses on predicting heart disease using machine learning by comparing five classification models. The goal is to determine which model performs best using various evaluation metrics such as accuracy, precision, recall, F1-score, ROC AUC, and inference time.

---

## 📚 Project Overview

- Trained and evaluated five ML models on a structured heart disease dataset.
- Performed detailed metric-based comparison along with visualizations.
- Visual tools include confusion matrices, ROC curves, and performance bar graphs.
- Dataset Source: [Heart Disease Dataset – Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

---

## 🚀 Models Implemented

- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier  
- Support Vector Machine (SVM)  
- K-Nearest Neighbors (KNN)

---

## 📊 Evaluation Metrics

| Model               | Accuracy | Precision | Recall | F1 Score | ROC AUC | Avg Precision | Inference Time (s) |
|--------------------|----------|-----------|--------|----------|---------|----------------|---------------------|
| Random Forest       | 0.9268   | 0.8947    | 0.9714 | 0.9315   | 0.9767  | 0.9751         | 0.00485             |
| SVM                 | 0.9268   | 0.9167    | 0.9429 | 0.9296   | 0.9771  | 0.9682         | 0.00595             |
| KNN                 | 0.8634   | 0.8738    | 0.8571 | 0.8654   | 0.9629  | 0.9620         | 0.00715             |
| Decision Tree       | 0.8537   | 0.8378    | 0.8857 | 0.8611   | 0.8810  | 0.8313         | 0.00035             |
| Logistic Regression | 0.8098   | 0.7619    | 0.9143 | 0.8312   | 0.9298  | 0.9316         | 0.00025             |

---

## 🛠 Tools & Libraries

- **Language**: Python 3.x
- **IDE**: Jupyter Notebook
- **Libraries**:
  - pandas, numpy – Data handling
  - matplotlib, seaborn – Visualization
  - scikit-learn – Model training and evaluation
  - joblib – Model saving/loading
  - time – Inference time measurement

---

## 🧪 Project Files

- `Data Preprocessing.py` – Loads and scales the dataset
- `Data Training.py` – Trains and saves 5 different models
- `Data Evaluation.py` – Evaluates all models using metrics and plots
- `model_evaluation_results.csv` – Final comparison of metrics
- `heart.csv` – Input dataset

---

## 📈 Visual Outputs

The project includes:
- Confusion matrices for all models
- ROC & Precision-Recall curves
- Bar graphs for metric comparisons
- Feature importance plots (where applicable)

> 📂 All graphs are included in the `screenshots/` folder.

---

## 🏁 Result Summary

While both **Random Forest** and **SVM** achieved the highest accuracy (92.68%), Random Forest performed slightly better in recall and F1-score, making it the most balanced and effective model in this analysis.

---

## 🧠 Future Work

- Implement hyperparameter tuning (GridSearchCV)
- Add k-fold cross-validation
- Develop a web interface using Streamlit or Flask
- Test on a larger or real-time clinical dataset

---

