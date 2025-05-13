"""
There's no outliers found in the dataset.
This script is used to detect outliers in the dataset using Z-score and IQR methods.
It also visualizes the distribution of the data and the outliers using histograms, boxplots, and Q-Q plots.
It uses the Shapiro-Wilk test to check for normality of the data.
"""

import polars as pl
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy import stats

# Use scan for large files
df = pl.scan_parquet("../../data/processed/loan_data_processed_v1.parquet")

essential_numerical_cols = ['loan_amnt', 'int_rate', 'installment', 'annual_inc']

# Preview
# print(df)
# print("\nData types:\n", df.schema)

# Aggregate data by purpose and calculate mean
for column in essential_numerical_cols:
    agg_df = (
        df.group_by('purpose')
        .agg(pl.col(column).mean())
        .collect()
        .to_pandas()
    )


    # 
    # 
    # Distribution plot with enhanced normality tests
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Histogram with KDE
    sns.histplot(data=agg_df, x=column, kde=True, ax=ax1)
    ax1.set_title(f'Distribution of {column.replace("_", " ").title()}')
    ax1.set_xlabel(column.replace("_", " ").title())
    ax1.set_ylabel('Count')

    # Q-Q plot to check normality
    stats.probplot(agg_df[column], dist="norm", plot=ax2)
    ax2.set_title("Q-Q Plot")

    # Boxplot
    sns.boxplot(x=agg_df[column], ax=ax3)
    ax3.set_title(f'Boxplot of {column.replace("_", " ").title()}')

    # Add statistical tests results
    # Skewness measures the asymmetry of the distribution
    # Positive skew = longer tail on right side, negative skew = longer tail on left side
    # Values close to 0 indicate symmetrical distribution
    skewness = stats.skew(agg_df[column])

    # Kurtosis measures "tailedness" of distribution compared to normal distribution
    # Positive kurtosis = heavier tails, negative = lighter tails than normal distribution
    # Normal distribution has kurtosis of 0
    kurtosis = stats.kurtosis(agg_df[column])

    # Shapiro-Wilk test checks if data is normally distributed
    # Null hypothesis is that data is normally distributed
    # If p-value < 0.05, reject null hypothesis (data is not normal)
    # If p-value > 0.05, fail to reject null hypothesis (data may be normal)
    # If Statistics value close to 1 indicate the data is more normally distributed
    # If Statistics value closer to 0 indicate stronger deviation from normality
    shapiro_stat, shapiro_p = stats.shapiro(agg_df[column])
    
    stats_text = (f'Skewness: {skewness:.3f}\n'
                 f'Kurtosis: {kurtosis:.3f}\n\n'
                 f'Shapiro-Wilk test:\n'
                 f'Statistic: {shapiro_stat:.3f}\n'
                 f'p-value: {shapiro_p:.3e}')
    
    ax4.text(0.1, 0.5, stats_text, fontsize=10)
    ax4.set_axis_off()

    plt.tight_layout()
    plt.show()


    # 
    # 
    # Apply Z-score Method to Detect Outliers if Dataset is Normal:
    
        # 1. Your data appears to follow a normal distribution
        # 2. Z-score is best suited for normally distributed data
        # 3. Using threshold of 3 standard deviations (>3 or <-3) captures 99.7% of normal data
        # Alternative methods like IQR could be used for non-normal distributions  
        # 4. Z-score method is sensitive to extreme values, so consider using IQR or other robust methods if data is skewed or has outliers


    # 
    # 
    # Calculate z-scores
    # agg_df['z_score'] = stats.zscore(agg_df[column])
    # outliers = agg_df[agg_df['z_score'].abs() > 3]
    # non_outliers = agg_df[agg_df['z_score'].abs() <= 3]
    
    
    # # 
    # # 
    # # Scatter plot
    # # Plot outliers in red
    # # Plot non-outliers in blue
    # plt.figure(figsize=(10, 6))
    
    # # Plot non-outliers
    # plt.scatter(non_outliers.index, 
    #             non_outliers[column],
    #             c='blue', 
    #             label='Normal Values')
    
    # # Plot outliers
    # plt.scatter(outliers.index,
    #             outliers[column],
    #             c='red',
    #             marker='x',
    #             s=100,
    #             label='Outliers')
    
    # plt.title(f'Outliers in {column.replace("_", " ").title()} (Z-score method)')
    # plt.xlabel('Purpose')
    # plt.ylabel(column.replace('_', ' ').title())
    # plt.xticks(rotation=45)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    
    # # Print outlier details
    # if len(outliers) > 0:
    #     print(f"\nOutliers detected in {column}:")
    #     print(outliers)

    # # 
    # # 
    # # Histogram with KDE for normal data and outliers
    # plt.figure(figsize=(12, 6))

    # sns.histplot(agg_df[column], bins=50, kde=True, color='#0066ff', alpha=0.6, label="Normal Data")  
    # sns.histplot(outliers[column], bins=30, kde=True, color='#E63946', alpha=0.8, label="Outliers")

    # plt.title(f'Distribution of {column.replace("_", " ").title()} with Outliers (Z-score Method)', fontsize=14, pad=10, color='#2F2F2F')
    # plt.xlabel(f'{column.replace("_", " ").title()} ($)', fontsize=12)
    # plt.ylabel('Frequency', fontsize=12)
    # plt.legend()

    # # Add text annotations for outliers
    # for idx, row in outliers.iterrows():
    #     plt.annotate(f'Purpose: {idx}\nValue: {row[column]:.2f}',
    #                 xy=(row[column], 0),
    #                 xytext=(10, 10),
    #                 textcoords='offset points',
    #                 ha='left',
    #                 va='bottom',
    #                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
    #                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    # plt.grid(True, linestyle='--', alpha=0.5)
    # plt.gca().set_facecolor('white')
    # plt.show()


    # 
    # 
    # Apply IQR Method to Detect Outliers if Dataset is Skewed:
    # Q1 = agg_df[column].quantile(0.25)
    # Q3 = agg_df[column].quantile(0.75)
    # IQR = Q3 - Q1

    # upper_bound = Q3 + 1.5 * IQR
    # lower_bound = Q1 - 1.5 * IQR

    # outliers = agg_df[(agg_df[column] < lower_bound) | (agg_df[column] > upper_bound)]
    # print(outliers.describe())

    # plt.figure(figsize=(12, 6))
    # sns.histplot(agg_df[column], bins=50, kde=True, color='#0066ff', alpha=0.6, label="Normal Data")  
    # sns.histplot(outliers[column], bins=30, kde=True, color='#E63946', alpha=0.8, label="Outliers")  

    # plt.title(f'Distribution of {column.replace("_", " ").title()} with Outliers (IQR Method)', fontsize=14, pad=10, color='#2F2F2F')
    # plt.xlabel(f'{column.replace("_", " ").title()} ($)', fontsize=12)
    # plt.ylabel('Frequency', fontsize=12)
    # plt.legend()

    # plt.grid(True, linestyle='--', alpha=0.5)
    # plt.gca().set_facecolor('white')

    # plt.title(f'Distribution of {column.replace("_", " ").title()} with Outliers (IQR Method)', fontsize=14, pad=10, color='#2F2F2F')
    # plt.xlabel(f'{column.replace("_", " ").title()} ($)', fontsize=12)
    # plt.ylabel('Frequency', fontsize=12)
    # plt.legend()

    # plt.grid(True, linestyle='--', alpha=0.5)
    # plt.gca().set_facecolor('white')
    # plt.show()
