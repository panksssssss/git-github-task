import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("cleaned_agricultural_data.csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

print("Dataset loaded successfully!")
print(df.head())

# 1. Average Production by Crop
plt.figure(figsize=(10, 6))

df.groupby("Crop")["Production_Tonnes"].mean().plot(kind="bar")

plt.title("Average Production by Crop")
plt.xlabel("Crop")
plt.ylabel("Average Production (Tonnes)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 2. Average Production by Season
plt.figure(figsize=(8, 5))

df.groupby("Season")["Production_Tonnes"].mean().plot(kind="bar")

plt.title("Average Production by Season")
plt.xlabel("Season")
plt.ylabel("Average Production (Tonnes)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# 3. Farm Area vs Production
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Farm_Area_Hectares",
    y="Production_Tonnes"
)

plt.title("Farm Area vs Production")
plt.xlabel("Farm Area (Hectares)")
plt.ylabel("Production (Tonnes)")
plt.tight_layout()
plt.show()


# 4. Rainfall vs Production
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Rainfall_mm",
    y="Production_Tonnes"
)

plt.title("Rainfall vs Production")
plt.xlabel("Rainfall (mm)")
plt.ylabel("Production (Tonnes)")
plt.tight_layout()
plt.show()


# 5. Production Distribution
plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Production_Tonnes",
    bins=20,
    kde=True
)

plt.title("Distribution of Agricultural Production")
plt.xlabel("Production (Tonnes)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# 6. Temperature Distribution
plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Avg_Temperature_C",
    bins=20,
    kde=True
)

plt.title("Distribution of Average Temperature")
plt.xlabel("Temperature (°C)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# 7. Correlation Matrix
numeric_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

correlation = df[numeric_columns].corr()

plt.figure(figsize=(14, 10))

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


print("All 7 visualizations completed successfully!")
