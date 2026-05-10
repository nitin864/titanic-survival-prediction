
# Titanic Survival Prediction Project

This project focuses on predicting passenger survival on the Titanic dataset using different machine learning models.

The project includes:
- Feature engineering
- Model comparison
- Hyperparameter tuning
- Cross validation
- Error analysis
- Model saving using `.pkl`
- Result saving using `.json`

---

# Project Structure

```

├── cleaned_data/
│   └── titanic_featured.csvmodel_optimization/
│
├── notebooks/
│   ├── cross_validation.ipynb
│   ├── error_analysis.ipynb
│   ├── feature_engineering_advanced.ipynb
│   ├── hyperparameter_tuning.ipynb
│   └── model_comparison_persistence.ipynb
│
│
├── results/
│   ├── cross_validation_results.json
│   ├── error_analysis_results.json
│   ├── hyperparameter_results.json
│   ├── model_comparison.json
│   │
│   ├── cross_validation_model.pkl
│   ├── error_analysis_model.pkl
│   ├── hyperparameter_model.pkl
│   ├── logistic_regression.pkl
│   └── random_forest.pkl
│
└── README.md

```

---

# Libraries Used

- pandas
- numpy
- scikit-learn
- joblib
- json

---

# Models Used

- Logistic Regression
- Random Forest Classifier

---

# Main Tasks Performed

## 1. Feature Engineering
Created new features such as:
- FamilySize
- FarePerPerson
- IsChild
- IsRich

## 2. Hyperparameter Tuning
Used `GridSearchCV` to find better model parameters.

## 3. Cross Validation
Performed 5-fold cross validation for checking model stability.

## 4. Error Analysis
Used:
- confusion matrix
- classification report

for model evaluation.

## 5. Model Persistence
Saved trained models using `.pkl` files.

---

# How to Run

1. Open Jupyter Notebook
2. Run notebooks one by one
3. Results will automatically be saved inside the `results/` folder

---

# Output Files

The project generates:
- JSON result files
- Trained model `.pkl` files
- Feature engineered datasets

---

# Author

Nitin Raj
