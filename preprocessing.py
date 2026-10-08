import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

# Load the Titanic dataset from seaborn (or from URL if seaborn not available)
try:
    df = sns.load_dataset('titanic')
except:
    # Fallback to URL
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)

print("Dataset loaded successfully.")
print(f"Shape: {df.shape}")

# 1. Explore basic info
print("\n=== Basic Info ===")
print(df.info())
print("\n=== First 5 rows ===")
print(df.head())
print("\n=== Missing values ===")
print(df.isnull().sum())
print("\n=== Data types ===")
print(df.dtypes)

# 2. Handle missing values
# We'll handle numerical and categorical columns separately
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"\nNumerical columns: {num_cols}")
print(f"Categorical columns: {cat_cols}")

# For numerical columns, fill missing values with median (robust to outliers)
for col in num_cols:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"Filled missing values in {col} with median: {median_val}")

# For categorical columns, fill missing values with mode
for col in cat_cols:
    if df[col].isnull().any():
        mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
        df[col].fillna(mode_val, inplace=True)
        print(f"Filled missing values in {col} with mode: {mode_val}")

# 3. Convert categorical features into numerical
# We'll use LabelEncoder for binary categorical and one-hot encoding for multi-class
# But for simplicity, we'll use one-hot encoding for all categorical columns (drop first to avoid multicollinearity)
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

print(f"\nAfter encoding, shape: {df_encoded.shape}")

# 4. Normalize/standardize the numerical features
# We'll standardize (zero mean, unit variance) the numerical columns (excluding the target if any)
# Let's assume we are not predicting, just preprocessing. We'll scale all numerical columns.
# But note: after one-hot encoding, we have new numerical columns (0,1). We should not scale those.
# So we'll scale only the original numerical columns (that were not encoded).

# Identify original numerical columns (before encoding)
original_num_cols = [col for col in num_cols if col in df_encoded.columns]

# Initialize scaler
scaler = StandardScaler()
df_scaled = df_encoded.copy()
df_scaled[original_num_cols] = scaler.fit_transform(df_encoded[original_num_cols])

print(f"\nAfter scaling, shape: {df_scaled.shape}")

# 5. Visualize outliers using boxplots and remove them
# We'll visualize outliers for the original numerical columns (before encoding) after scaling
# But note: scaling changes the distribution, but outliers remain outliers in terms of IQR.

# We'll remove outliers using IQR method for each original numerical column
# We'll create a copy to remove outliers
df_clean = df_scaled.copy()

outlier_indices = []
for col in original_num_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)].index
    outlier_indices.extend(outliers)
    print(f"Column {col}: {len(outliers)} outliers detected")

# Remove duplicates (if an index is outlier in multiple columns)
outlier_indices = list(set(outlier_indices))
print(f"\nTotal unique outlier indices: {len(outlier_indices)}")

# Remove outliers
df_clean = df_clean.drop(index=outlier_indices)
print(f"After removing outliers, shape: {df_clean.shape}")

# Visualize boxplots before and after removing outliers (for one column as example)
# Let's choose 'fare' if exists, else first numerical column
if 'fare' in df.columns:
    plot_col = 'fare'
elif original_num_cols:
    plot_col = original_num_cols[0]
else:
    plot_col = None

if plot_col:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Before removing outliers
    sns.boxplot(y=df[plot_col], ax=axes[0])
    axes[0].set_title(f'Boxplot of {plot_col} (Before removing outliers)')
    # After removing outliers (note: we need to use the scaled version for after? 
    # But for visualization, we can show the original values before scaling after removing the same indices)
    # Let's get the original values of the non-outlier rows
    non_outlier_vals = df.loc[df_clean.index, plot_col]
    sns.boxplot(y=non_outlier_vals, ax=axes[1])
    axes[1].set_title(f'Boxplot of {plot_col} (After removing outliers)')
    plt.tight_layout()
    plt.savefig('outlier_boxplots.png')
    plt.close()
    print(f"Saved boxplot comparison for {plot_col} as 'outlier_boxplots.png'")

# Save the processed data
df_clean.to_csv('titanic_processed.csv', index=False)
print("\nProcessed data saved to 'titanic_processed.csv'")

# Also save the original dataset for reference
df.to_csv('titanic_original.csv', index=False)
print("Original data saved to 'titanic_original.csv'")

print("\n=== Task completed successfully ===")
