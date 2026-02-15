import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("titanic.csv")

print("\n---First 5_rows from Titanic Data---")
print(df.head())

print("\n---Data Information from Titanic Data---")
print(df.info())

print("\n---Missing Age Information---")
missing_age =df[df["Age"].isnull()]
print(missing_age.head())

missing_age_count = df["Age"].isnull().sum()
print(f"\nMissing Age Count: {missing_age_count} passengers")

print("\n---Fixing Missing Age Data From Titanic---")
fix_age_data =df["Age"].median()
df["Age"] = df["Age"].fillna(fix_age_data)
print(fix_age_data)
print(f"After fixing Age Count: {df["Age"].isnull().sum()}")

print("\n---Removing Missing Cabin---")
df = df.drop(columns= ["Cabin"])
print(df.info())

print("\n---Saving Titanic Clean Data---")
df.to_csv("after_fixing_titanic_data.csv", index=False)
print("File Saved Successfully!")


plt.figure(figsize=(6,4))
sns.histplot(data=df, x="Age", hue="Survived", kde=True, element="step")
plt.title("Titanic: Age Distribution by Survival Status", fontsize=16, fontweight='bold')
plt.xlabel("Age (Years)")
plt.ylabel("Count")
plt.show()

print("\n---Calculating Survived Age---")
calculating_age_survived = df.groupby("Sex")["Survived"].mean()
print(calculating_age_survived)


df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

plt.figure(figsize=(6,4))
sns.barplot(data= df, x ="FamilySize", y="Survived", palette="viridis")
plt.title("Survived Rate by Family Size")
plt.xlabel("Family Size (SibSp + Parch)")
plt.ylabel("Survival Probability ( 0 to 1)")
plt.grid(axis="y")
plt.show()