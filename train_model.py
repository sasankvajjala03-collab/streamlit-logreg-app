import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import pickle

# Load data
train_df = pd.read_csv("Titanic_train.csv")

# Drop unused columns
train_df = train_df.drop(["Name", "Ticket", "Cabin"], axis=1)

# Fill missing values
train_df["Age"] = train_df["Age"].fillna(train_df["Age"].median())
train_df["Fare"] = train_df["Fare"].fillna(train_df["Fare"].median())
train_df["Embarked"] = train_df["Embarked"].fillna(train_df["Embarked"].mode()[0])

# Label Encoders
le_sex = LabelEncoder()
le_embarked = LabelEncoder()

train_df["Sex"] = le_sex.fit_transform(train_df["Sex"])
train_df["Embarked"] = le_embarked.fit_transform(train_df["Embarked"])

# Prepare data
X = train_df.drop(["Survived"], axis=1)
y = train_df["Survived"]

# Train model
model = LogisticRegression(max_iter=500)
model.fit(X, y)

# Save model & encoders
pickle.dump(model, open("titanic_model.pkl", "wb"))
pickle.dump(le_sex, open("le_sex.pkl", "wb"))
pickle.dump(le_embarked, open("le_embarked.pkl", "wb"))

print("✔ Model & encoders saved successfully!")
