import polars as pl
import numpy as np

df = pl.scan_parquet("../../data/raw/loan_data.parquet")



# Standardize column names:
new_names = {
    'loan_amnt': 'loan_amount',
    'term': 'loan_term_months',
    'int_rate': 'interest_rate',
    'installment': 'monthly_installment',
    'grade': 'loan_grade',
    'emp_length': 'employment_length_years',
    'home_ownership': 'home_ownership_status',
    'annual_inc': 'annual_income',
    'verification_status': 'income_verification_status',
    'purpose': 'loan_purpose',
    'addr_state': 'borrower_address_state',
    'dti': 'debt_to_income_ratio',
    'delinq_2yrs': 'delinquencies_last_2_years',
    'earliest_cr_line': 'earliest_credit_line_date',
    'fico_range_low': 'fico_score_lower_range',
    'fico_range_high': 'fico_score_upper_range',
    'inq_last_6mths': 'credit_inquiries_last_6_months',
    'open_acc': 'number_of_open_credit_accounts',
    'pub_rec': 'number_of_derogatory_public_records',
    'revol_bal': 'total_revolving_credit_balance',
    'revol_util': 'revolving_credit_utilization_rate',
    'total_acc': 'total_number_of_credit_accounts',
    'initial_list_status': 'initial_loan_listing_status',
    'collections_12_mths_ex_med': 'collections_last_12_months_excluding_medical',
    'application_type': 'loan_application_type',
    'acc_now_delinq': 'number_of_accounts_currently_delinquent',
    'tot_coll_amt': 'total_collection_amounts_ever',
    'tot_cur_bal': 'total_current_balance_all_accounts',
    'pub_rec_bankruptcies': 'number_of_public_record_bankruptcies',
    'tax_liens': 'number_of_tax_liens'
}

df = df.rename(new_names)
# print(df.collect().columns)



# Handle duplicates:
df = df.unique()
# print(df.collect().is_duplicated().sum())



# Handle missing data:
numerical_cols = [col for col, dtype in df.collect_schema().items() if dtype in [pl.Int64, pl.Float64]]
# print("Numerical columns:", numerical_cols)

all_cols = list(df.collect_schema().keys())
non_numerical_cols = [col for col in all_cols if col not in numerical_cols]
# print("Non-numerical columns:", non_numerical_cols)

# Fill missing value in debt_to_income_ratio with calculated value
# debt_to_income_ratio = (loan_amount / annual_income) * 100
df = df.with_columns(
    pl.when(pl.col("debt_to_income_ratio").is_null())
    .then(
        pl.when((pl.col("annual_income") != 0) & (pl.col("annual_income").is_not_null()) & (pl.col("loan_amount").is_not_null()))
        .then((pl.col("loan_amount") / pl.col("annual_income")) * 100)
        .otherwise(pl.lit(0.0))
    )
    .otherwise(pl.col("debt_to_income_ratio"))
    .alias("debt_to_income_ratio")
)
# print(df.collect().select("debt_to_income_ratio").null_count().sum())

# Fill missing values in numerical columns with 0
df = df.with_columns(
    [pl.col(col).fill_null(np.nan).alias(col) for col in numerical_cols]
)

# Converting data type of this date column from string to date
df = df.with_columns(
    pl.col("earliest_credit_line_date").str.strptime(pl.Date, "%b-%Y").alias("earliest_credit_line_date")
)
# print(df.collect_schema()["earliest_credit_line_date"])
# print(df.select("earliest_credit_line_date").collect().null_count().sum())

# Fill missing values in non-numerical columns with "NA"
df = df.with_columns(
    [pl.col(col).fill_null("NA").alias(col) for col in non_numerical_cols if col != "earliest_credit_line_date"]
)
# print(df.collect().null_count().sum())



# Handle incorrect data types:
potential_categorical_columns = [
    'loan_term_months', 'loan_grade', 'home_ownership_status', 'income_verification_status', 'loan_status',
    'loan_purpose', 'borrower_address_state', 'initial_loan_listing_status', 'loan_application_type'
]

# Function to check the number of unique values relative to the total rows
# and determine if a column should be categorical
def should_be_categorical(ldf: pl.LazyFrame, column_name: str, threshold_ratio: float = 0.05, min_unique: int = 2, sample_size: int = 1000):
    """
    Checks if a Polars Series in a LazyFrame should be converted to Categorical based on
    an estimated ratio of unique values to the total number of rows.
    """
    try:
        # Estimate unique count using a sample
        unique_count = ldf.select(pl.col(column_name).n_unique()).limit(sample_size).collect().item()
        total_rows = ldf.select(pl.len()).collect().item()

        if total_rows > 0 and (unique_count / total_rows) < threshold_ratio and unique_count >= min_unique:
            return True
        return False
    except pl.exceptions.ColumnNotFoundError:
        print(f"Warning: Column '{column_name}' not found in the LazyFrame.")
        return False

columns_to_categorize = []

# Identify columns to convert based on the criteria
for col in potential_categorical_columns:
    if col in df.collect().columns:
        if should_be_categorical(df, col):
            columns_to_categorize.append(col)
        else:
            print(f"Column '{col}' has too many unique values to be effectively categorical.")
    else:
        print(f"Warning: Column '{col}' not found in the DataFrame.")

# print("\nPotential columns to convert to Categorical:", columns_to_categorize)

# Change the data types to pl.Categorical
df_categorical = df.with_columns(
    [pl.col(col).cast(pl.Categorical).alias(col) for col in columns_to_categorize]
)
# print(df_categorical.collect().dtypes)


# Validate and verify data:
print("Null values:\n", df.collect().null_count().sum())
print("\nDuplicates:\n", df.collect().is_duplicated().sum())
print("\nCleaned data shape:\n", df.collect().shape)
print("\nData types:\n", df.collect_schema())


# Export processed data:
df.sink_parquet("../../data/processed/loan_data_processed_v1.parquet", compression="snappy", statistics=True, row_group_size=100000)
print("Data exported successfully!")
