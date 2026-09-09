import pandas as pd

df = pd.read_csv("seasonal_agriculture_performance_dataset.csv")


print("FIRST 5 RECORDS:")
print(df.head())


print("\nLAST 5 RECORDS:")
print(df.tail())


print("\nDATASET INFORMATION:")
df.info()


print("\nDATASET SHAPE:")
print(df.shape)


print("\nCOLUMN NAMES:")
print(df.columns.tolist())


print("\nDATA TYPES:")
print(df.dtypes)


print("\nMISSING VALUES:")
print(df.isnull().sum())


print("\nNUMBER OF DUPLICATES:")
print(df.duplicated().sum())


df = df.drop_duplicates()


numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())


categorical_columns = df.select_dtypes(include=["object"]).columns

for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])


print("\nMISSING VALUES AFTER CLEANING:")
print(df.isnull().sum())


print("\nDUPLICATES AFTER CLEANING:")
print(df.duplicated().sum())


print("\nCLEANED DATASET:")
print(df.head())


df.to_csv("cleaned_seasonal_agriculture_performance_dataset.csv", index=False)

print("\nCleaned dataset saved successfully.")
