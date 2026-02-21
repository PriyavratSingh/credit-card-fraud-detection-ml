# Credit Card Fraud Detection using Machine Learning

An end-to-end machine learning pipeline to detect fraudulent credit card transactions using a highly imbalanced real-world dataset.

---

## 🚀 Features

- Data preprocessing and feature scaling
- Handling class imbalance using class weights
- Random Forest classifier
- Model evaluation (Precision, Recall, F1-score, ROC-AUC)
- Model serialization using Pickle
- Streamlit-based real-time prediction interface

---

## 📊 Dataset

- 284,807 transactions
- 492 fraud cases (~0.17% fraud rate)
- Highly imbalanced dataset

⚠ Dataset is not included in this repository due to size limitations.

Download from Kaggle:
Credit Card Fraud Detection – Machine Learning Group (ULB)

---

## 🧠 Model Performance (Fraud Class)

- Precision: 96%
- Recall: 74%
- ROC-AUC: 0.87+

The model balances fraud detection capability while minimizing false positives.

---

## 📂 Project Structure

```
credit-card-fraud-detection-ml/
│
├── data/ (not included)
├── models/
│   ├── random_forest.pkl
│   └── scaler.pkl
├── notebooks/
│   └── eda_and_modeling.ipynb
├── src/
│   ├── train.py
│   └── predict.py
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙ Installation

```bash
git clone <your-repo-link>
cd credit-card-fraud-detection-ml
pip install -r requirements.txt
```

---

## 🏋 Train the Model

```bash
python src/train.py
```

---

## 🖥 Run the Streamlit App

```bash
streamlit run app.py
```

---

## 📌 Future Improvements

- SMOTE oversampling
- Hyperparameter tuning
- Cross-validation
- SHAP explainability
- Model monitoring

---

## 👤 Author

Priyavrat Singh