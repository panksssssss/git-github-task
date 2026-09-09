import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv("cleaned_agricultural_data.csv")

print("========== DATASET ==========")
print(df.head())


print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== DATASET INFORMATION ==========")
df.info()



print("\n========== DESCRIPTIVE STATISTICS ==========")
print(df.describe())



print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())



print("\n========== DUPLICATE RECORDS ==========")
print(df.duplicated().sum())



print("\n========== CROP COUNTS ==========")
if "Crop" in df.columns:
    print(df["Crop"].value_counts())

print("\n========== SEASON COUNTS ==========")
if "Season" in df.columns:
    print(df["Season"].value_counts())



numeric_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

correlation = df[numeric_columns].corr()

print("\n========== CORRELATION MATRIX ==========")
print(correlation)

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Matrix of Agricultural Variables")
plt.tight_layout()
plt.show()



if "Crop" in df.columns and "Production" in df.columns:

    crop_production = df.groupby("Crop")["Production"].mean()

    print("\n========== AVERAGE PRODUCTION BY CROP ==========")
    print(crop_production)

    crop_production.plot(
        kind="bar",
        figsize=(10, 6)
    )

    plt.title("Average Production by Crop")
    plt.xlabel("Crop")
    plt.ylabel("Average Production")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



if "Season" in df.columns and "Production" in df.columns:

    season_production = df.groupby("Season")["Production"].mean()

    print("\n========== AVERAGE PRODUCTION BY SEASON ==========")
    print(season_production)

    season_production.plot(
        kind="bar",
        figsize=(8, 5)
    )

    plt.title("Average Production by Season")
    plt.xlabel("Season")
    plt.ylabel("Average Production")
    plt.tight_layout()
    plt.show()



if "Farm_Area" in df.columns and "Production" in df.columns:

    plt.figure(figsize=(8, 5))

    sns.scatterplot(
        data=df,
        x="Farm_Area",
        y="Production"
    )

    plt.title("Farm Area vs Production")
    plt.xlabel("Farm Area")
    plt.ylabel("Production")
    plt.tight_layout()
    plt.show()



for column in numeric_columns:

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        y=df[column]
    )

    plt.title("Outlier Detection - " + column)
    plt.ylabel(column)
    plt.tight_layout()
    plt.show()



print("\n========== EDA COMPLETED ==========")
print("Dataset inspection completed.")
print("Descriptive statistics calculated.")
print("Missing values checked.")
print("Duplicate records checked.")
print("Correlation analysis completed.")
print("Patterns and trends identified.")
print("Outlier detection completed.")
