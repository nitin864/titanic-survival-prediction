# Titanic Survival Prediction — Machine Learning Project

A complete end-to-end machine learning project built on the Titanic dataset.
The project covers data preprocessing, exploratory data analysis, feature engineering,
model training, model comparison, and an interactive Streamlit dashboard.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [File Structure](#file-structure)
- [Notebooks](#notebooks)
- [Feature Engineering](#feature-engineering)
- [Models](#models)
- [Results](#results)
- [Dashboard](#dashboard)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [WakaTime](#wakatime)

---

## Project Overview

The goal of this project is to predict whether a passenger survived the Titanic disaster
based on features such as age, gender, passenger class, fare, and family size.

The project is divided into four main stages:

1. Data cleaning and preprocessing
2. Feature engineering
3. Model training and evaluation (Logistic Regression and Random Forest)
4. Interactive dashboard built with Streamlit

---

## Dataset

- Source: [Titanic Dataset — datasciencedojo](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)
- Total rows: 891 passengers
- Target column: `Survived` (1 = survived, 0 = did not survive)

Original columns used:

| Column | Description |
|---|---|
| Pclass | Passenger class (1, 2, 3) |
| Sex | Gender (encoded: male=1, female=0) |
| Age | Age in years |
| SibSp | Number of siblings or spouses aboard |
| Parch | Number of parents or children aboard |
| Fare | Ticket fare |
| Embarked | Port of embarkation (S=0, C=1, Q=2) |

Dropped columns: `PassengerId`, `Name`, `Ticket`, `Cabin`

---

## File Structure

```
task2/
│
├── cleaned_data/
│   ├── titanic_cleaned.csv         # Dataset after cleaning and encoding
│   └── titanic_featured.csv        # Dataset after feature engineering
│
├── notebooks/
│   │
│   ├── preprocessing/
│   │   └── preprocessing.ipynb     # EDA, cleaning, encoding, saving cleaned CSV
│   │
│   ├── featured_eng/
│   │   └── featured_engineering.ipynb  # Creates 5 new features, visualizes them
│   │
│   └── models/
│       ├── logistic_regression.ipynb   # Trains and evaluates Logistic Regression
│       ├── random_forest.ipynb         # Trains and evaluates Random Forest
│       │
│       ├── comparison/
│       │   └── model_comparison.ipynb  # Side-by-side comparison of both models
│       │
│       └── results/
│           ├── logistic_regression.json    # Saved metrics from LR model
│           └── random_forest.json          # Saved metrics from RF model
│
├── dashboard.py        # Streamlit interactive dashboard
├── requirements.txt    # Python dependencies
└── README.md
```

---

## Notebooks

### 1. preprocessing.ipynb

Located at: `notebooks/preprocessing/preprocessing.ipynb`

This notebook handles the full data cleaning pipeline:

- Loads the Titanic dataset directly from a public URL
- Converts the `Survived` column to yes/no labels for readability during EDA
- Checks data types, null counts, and missing value percentages
- Visualizes missing values using a seaborn heatmap
- Performs EDA with charts: survival count, age distribution, fare boxplot,
  survival by gender, survival by passenger class
- Handles missing values:
  - `Age` filled with median
  - `Embarked` filled with mode
  - `Cabin` dropped (77% missing)
- Encodes categorical columns: Sex (male=1, female=0), Embarked (S=0, C=1, Q=2)
- Drops irrelevant columns: PassengerId, Name, Ticket, Cabin
- Plots a correlation heatmap on the cleaned data
- Saves the cleaned dataset to `cleaned_data/titanic_cleaned.csv`

---

### 2. featured_engineering.ipynb

Located at: `notebooks/featured_eng/featured_engineering.ipynb`

This notebook creates five new features from the cleaned dataset to improve model performance:

| Feature | How it is calculated | Why it helps |
|---|---|---|
| Family_Size | SibSp + Parch + 1 | Captures total family size aboard |
| is_alone | 1 if Family_Size == 1 else 0 | Isolates passengers traveling alone |
| Age_x_Pclass | Age multiplied by Pclass | Captures interaction between age and class |
| FarePerPerson | Fare divided by Family_Size | Normalizes fare across group sizes |
| AgeGroup | Age bucketed: child/teen/adult/senior | Groups age into meaningful categories |

Each new feature is visualized against the Survived column to confirm relevance.

The enriched dataset is saved to `cleaned_data/titanic_featured.csv`.

---

### 3. logistic_regression.ipynb

Located at: `notebooks/models/logistic_regression.ipynb`

Steps:
- Loads `titanic_featured.csv`
- Splits data: 80% training, 20% testing with stratify=y and random_state=42
- Fills any remaining NaN values using `SimpleImputer(strategy='mean')`
- Normalizes features using `StandardScaler`
- Trains `LogisticRegression(max_iter=200, random_state=42)`
- Evaluates using: Accuracy, ROC-AUC, Classification Report
- Plots: Confusion Matrix and ROC Curve
- Saves results to `results/logistic_regression.json`

---

### 4. random_forest.ipynb

Located at: `notebooks/models/random_forest.ipynb`

Steps:
- Loads `titanic_featured.csv`
- Same train/test split, imputation, and scaling as above
- Trains `RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)`
- Evaluates using: Accuracy, ROC-AUC, Cross-Validation Score (cv=5), Classification Report
- Plots: Confusion Matrix, ROC Curve, and Feature Importance bar chart
- Saves results to `results/random_forest.json`

---

### 5. model_comparison.ipynb

Located at: `notebooks/models/comparison/model_comparison.ipynb`

- Loads saved JSON results from both models
- Builds a side-by-side metrics table
- Plots accuracy comparison bar chart
- Plots ROC-AUC comparison bar chart
- Determines which model performed better and why

---

## Feature Engineering

The five engineered features and their logic:

```python
# Family size — total people aboard including self
df['Family_Size'] = df['SibSp'] + df['Parch'] + 1

# Is alone — 1 if traveling solo, 0 otherwise
df['is_alone'] = (df['Family_Size'] == 1).astype(int)

# Age x Pclass — interaction term
df['Age_x_Pclass'] = df['Age'] * df['Pclass']

# Fare per person — actual cost per individual in the group
df['FarePerPerson'] = df['Fare'] / df['Family_Size']

# Age group — binned age categories
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100],
                         labels=[0, 1, 2, 3, 4]).astype(int)
```

---

## Models

### Logistic Regression

A linear model that estimates the probability of survival using a sigmoid function.
Chosen as the baseline model because it is interpretable and works well for binary classification.

Key settings: `max_iter=200`, `random_state=42`, features scaled with `StandardScaler`.

### Random Forest

An ensemble of 100 decision trees. Each tree is trained on a random subset of data
and features, and the final prediction is the majority vote across all trees.
Chosen as the second model because it captures non-linear relationships and is
resistant to overfitting.

Key settings: `n_estimators=100`, `max_depth=6`, `random_state=42`.

---

## Results

| Model | Accuracy | ROC-AUC | CV Score (5-fold) |
|---|---|---|---|
| Logistic Regression | 80.45% | 0.8563 | N/A |
| Random Forest | 79.33% | 0.8466 | 0.8189 |

Logistic Regression achieved slightly higher accuracy and ROC-AUC on this dataset.
Random Forest provides cross-validation score which shows consistent performance
across different data splits (0.8189), indicating it is not overfitting.

---

## Dashboard

The interactive dashboard is built with Streamlit and loaded from `dashboard.py`.

Features of the dashboard:

- Sidebar filters: Passenger Class, Gender, Age Range — all charts update live
- KPI cards: Total passengers, Survived count, Survival rate, Average age
- Survival by Gender — grouped bar chart
- Survival by Passenger Class — grouped bar chart
- Age Distribution by Survival — overlapping histogram
- Fare vs Age scatter plot — color-coded by survival
- Embarkation Port vs Survival — bar chart
- Family Size vs Survival Rate — bar chart
- Feature Correlation Heatmap — full seaborn heatmap
- Model Comparison — accuracy and ROC-AUC bar charts with results table
- Raw Data Table — expandable filtered data viewer

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebooks

Open notebooks in order using Google Colab or Jupyter:

```
1. notebooks/preprocessing/preprocessing.ipynb
2. notebooks/featured_eng/featured_engineering.ipynb
3. notebooks/models/logistic_regression.ipynb
4. notebooks/models/random_forest.ipynb
5. notebooks/models/comparison/model_comparison.ipynb
```

### 4. Run the Streamlit dashboard

```bash
streamlit run dashboard.py
```

Make sure `dashboard.py` is in the same directory as the `cleaned_data/` folder.

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## WakaTime

 

![WakaTime Dashboard](wakatime_screenshot.png)
