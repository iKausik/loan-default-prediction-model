import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# Load dataset
df = pl.scan_parquet("../../data/processed/loan_data_processed_v1.parquet")

# Select raw features (including those needed for engineered features)
raw_features = [
    "loan_amount",
    "monthly_installment",    
    "annual_income",
    "debt_to_income_ratio", 
    "revolving_credit_utilization_rate",
    "delinquencies_last_2_years",
    "credit_inquiries_last_6_months",
    "number_of_open_credit_accounts",
    "number_of_derogatory_public_records",
    "collections_last_12_months_excluding_medical",
    "number_of_accounts_currently_delinquent",
    "total_collection_amounts_ever",
    "number_of_public_record_bankruptcies",
    "number_of_tax_liens",
    "employment_length_years",
    # Raw columns for feature engineering
    "fico_score_lower_range",
    "fico_score_upper_range",
    "interest_rate",
    "income_verification_status",
    "loan_term_months",
    # Categorical columns to encode
    "loan_grade",
    "loan_purpose",
    "home_ownership_status",
    "initial_loan_listing_status",
    "loan_application_type",
    # Target
    "loan_status"
]

# Load selected columns lazily
lf = df.select(raw_features)

# Engineer features
lf = lf.with_columns([
    # Numeric interactions and transformations
    ((pl.col("fico_score_lower_range") + pl.col("fico_score_upper_range")) / 2).alias("fico_score_avg"),
    (pl.col("debt_to_income_ratio") ** 2).alias("debt_to_income_ratio_squared"),
    (pl.col("annual_income") / pl.col("loan_amount")).alias("income_per_loan_amount"),
    (pl.col("monthly_installment") / pl.col("annual_income")).alias("installment_to_income_ratio"),
    (pl.col("revolving_credit_utilization_rate") ** 2).alias("revolving_credit_utilization_rate_squared"),
    (pl.col("loan_amount") * pl.col("interest_rate")).alias("loan_amount_interest_interaction"),
    (pl.col("income_verification_status") != "Not Verified").cast(pl.Int8).alias("income_verified"),
    (pl.col("loan_term_months").str.extract(r"(\d+)").cast(pl.Float64) / 12).alias("loan_term_years"),
    (pl.when(pl.col("employment_length_years").str.strip_chars().str.to_lowercase() == "na").then(None).otherwise(pl.col("employment_length_years").str.extract(r"(\d+)", 1).cast(pl.Int8)).alias("employment_length_years"))
]).fill_nan(0)

# Drop raw columns not needed after engineering
lf = lf.drop(["fico_score_lower_range", "fico_score_upper_range", "interest_rate", 
             "income_verification_status", "loan_term_months"])

# One-hot encode categorical columns
for col in [
    "loan_grade",
    "loan_purpose", 
    "home_ownership_status", 
    "initial_loan_listing_status",
    "loan_application_type"
]:
    # Get unique values for the column
    unique_vals = lf.select(col).unique().collect()[col].to_list()
    # Create binary columns for each unique value
    lf = lf.with_columns([
        (pl.col(col) == val).cast(pl.Int8).alias(f"{col}_{val}")
        for val in unique_vals
    ])
    # Drop original column
    lf = lf.drop(col)

# Separate features and target
df_processed = lf

y = df_processed.select((pl.col("loan_status") == "Charged Off").cast(pl.Int8).alias("loan_status"))
X = df_processed.drop("loan_status")

# Collect to pandas for sklearn
X_pandas = X.collect().to_pandas()
y_pandas = y.collect().to_pandas()["loan_status"]

# Handle infinity or NaN values in the dataset
X_pandas.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace inf with NaN
X_pandas.fillna(0, inplace=True)  # Replace NaN with 0

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_pandas, y_pandas, test_size=0.2, random_state=42, stratify=y_pandas
)

# Identify numeric columns for scaling
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

# Preprocessing pipeline: scale numeric, passthrough dummies
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols)
], remainder="passthrough")

pipeline = Pipeline([("preprocessor", preprocessor)])
pipeline.fit(X_train)

# Transform data
X_train_processed = pipeline.transform(X_train)
X_test_processed = pipeline.transform(X_test)

# Convert back to Polars for inspection
new_feature_names = numeric_cols + [col for col in X_train.columns if col not in numeric_cols]
X_train_df = pl.DataFrame(X_train_processed, schema=new_feature_names)
X_test_df = pl.DataFrame(X_test_processed, schema=new_feature_names)

# print("Processed Training Data Shape:", X_train_df.shape)
# print("Processed Testing Data Shape:", X_test_df.shape)
# print("\nFirst few rows of processed training data:")
# for col in X_train_df.columns:
#     print(f"\n{col}:")
#     print(X_train_df.select(col).head().to_series().to_list())

# Export processed data function
def get_processed_data():
    return X_train_df, X_test_df, y_train, y_test
