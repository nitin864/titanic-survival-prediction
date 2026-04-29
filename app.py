# Titanic_dashboard
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Titanic Dashboard", layout="wide")

# load_data
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_data/titanic_featured.csv")

df = load_data()

# Convert encoded columns
df["Sex"] = df["Sex"].map({1: "Male", 0: "Female"})
df["Embarked"] = df["Embarked"].map({0: "Southampton", 1: "Cherbourg", 2: "Queenstown"})
df["Survived"] = df["Survived"].map({1: "Survived", 0: "Not Survived"})

# Sidebar filters
st.sidebar.title("Filters")

pclass = st.sidebar.multiselect("Class", [1, 2, 3], default=[1, 2, 3])
gender = st.sidebar.multiselect("Gender", ["Male", "Female"], default=["Male", "Female"])
age_range = st.sidebar.slider("Age", 0, 80, (0, 80))

filtered = df[
    (df["Pclass"].isin(pclass)) &
    (df["Sex"].isin(gender)) &
    (df["Age"].between(age_range[0], age_range[1]))
]

# Title
st.title("Titanic Survival Dashboard")

# Metrics
total = len(filtered)
survived = (filtered["Survived"] == "Survived").sum()
rate = round((survived / total) * 100, 1) if total else 0
avg_age = round(filtered["Age"].mean(), 1) if total else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total", total)
c2.metric("Survived", survived)
c3.metric("Survival %", f"{rate}%")
c4.metric("Avg Age", avg_age)

# Survival by gender
st.subheader("Survival by Gender")
g = filtered.groupby(["Sex", "Survived"]).size().unstack()
st.bar_chart(g)

# Survival by class
st.subheader("Survival by Class")
c = filtered.groupby(["Pclass", "Survived"]).size().unstack()
st.bar_chart(c)

# Age distribution
st.subheader("Age Distribution")
fig, ax = plt.subplots()
for label in ["Survived", "Not Survived"]:
    data = filtered[filtered["Survived"] == label]["Age"].dropna()
    ax.hist(data, bins=20, alpha=0.5, label=label)
ax.legend()
st.pyplot(fig)

# Scatter plot
st.subheader("Fare vs Age")
fig, ax = plt.subplots()
colors = filtered["Survived"].map({"Survived": "green", "Not Survived": "red"})
ax.scatter(filtered["Age"], filtered["Fare"], c=colors, alpha=0.5)
st.pyplot(fig)

# Heatmap
st.subheader("Correlation Heatmap")
cols = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
corr = filtered[cols].corr()

fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, ax=ax)
st.pyplot(fig)

#Model Comparison
st.subheader("Model Comparison")

model_data = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [0.80, 0.79],
    "ROC-AUC": [0.85, 0.84]
})

st.table(model_data.set_index("Model"))

# Accuracy chart
st.subheader("Accuracy Comparison")
fig, ax = plt.subplots()
ax.bar(model_data["Model"], model_data["Accuracy"])
ax.set_ylabel("Accuracy")
st.pyplot(fig)

# ROC-AUC chart
st.subheader("ROC-AUC Comparison")
fig, ax = plt.subplots()
ax.bar(model_data["Model"], model_data["ROC-AUC"])
ax.set_ylabel("ROC-AUC")
st.pyplot(fig)

# Raw data
with st.expander("Show Data"):
    st.dataframe(filtered)