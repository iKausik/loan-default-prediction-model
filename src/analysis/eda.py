import polars as pl

df = pl.scan_parquet("../../data/raw/loan_data.parquet")

# Explore the data:
print("First 5 rows of the dataset:")
print(df.head(5).collect())
print("\n")

print("Data information:")
print(df.collect_schema())
print("\n")

print("Data Shape:")
print(df.collect().shape)
print("\n")

print("Summary statistics:")
print(df.describe())
print("\n")

print("Missing values:")
print(df.collect().null_count().sum())
print("\n")

print("Duplicate rows:")
print(df.collect().is_duplicated().sum())
print("\n")
