import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Create a directory to store visualizations and reports
os.makedirs("updated_visualizations", exist_ok=True)
os.makedirs("updated_reports", exist_ok=True)

# Load datasets with error handling
try:
    train_data = pd.read_csv("train.csv")
    test_data = pd.read_csv("test.csv")
    sample_submission = pd.read_csv("sample_submission.csv")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Ensure train.csv, test.csv, and sample_submission.csv are in the working directory.")
    exit()

# Verify datasets are not empty
if train_data.empty or test_data.empty or sample_submission.empty:
    print("One or more datasets are empty. Please check the input files.")
    exit()

# Display basic dataset info
print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)
print("Sample submission shape:", sample_submission.shape)

# Function to calculate and save missing value details
def save_missing_values_report(df, dataset_name, filename_prefix):
    missing = df.isnull().sum()
    total = len(df)
    missing_percent = (missing / total) * 100
    missing_report = pd.DataFrame({
        "Feature": missing.index,
        "Missing Count": missing.values,
        "Missing Percent": missing_percent.values
    }).query("`Missing Count` > 0").sort_values(by="Missing Percent", ascending=False)

    if missing_report.empty:
        print(f"No missing values in {dataset_name}.")
    else:
        print(f"Missing values in {dataset_name}:")
        print(missing_report)
        # Save the report as a CSV file
        missing_report.to_csv(f"updated_reports/{filename_prefix}_missing_values.csv", index=False)

        # Plot missing values as a bar chart
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Missing Percent", y="Feature", data=missing_report, palette="coolwarm")
        plt.title(f"Missing Values in {dataset_name}", fontsize=16, weight='bold')
        plt.xlabel("Percentage of Missing Values", fontsize=12)
        plt.ylabel("Features", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"updated_visualizations/{filename_prefix}_missing_values.png")
        plt.close()

# Save missing value reports for train and test datasets
save_missing_values_report(train_data, "Train Dataset", "train_dataset")
save_missing_values_report(test_data, "Test Dataset", "test_dataset")

# Feature engineering
numerical_cols = ['Age', 'Work Experience', 'Family  Size']
for col in numerical_cols:
    if col in train_data.columns:
        train_data[col] = pd.to_numeric(train_data[col], errors='coerce')
    if col in test_data.columns:
        test_data[col] = pd.to_numeric(test_data[col], errors='coerce')

def add_features(df):
    if 'Age' in df.columns and 'Work Experience' in df.columns:
        df['Age_Group'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 55, 100],
                                 labels=['Young', 'Young_Adult', 'Adult', 'Senior', 'Elder'])
        df['Experience_Age_Ratio'] = df['Work Experience'] / df['Age']
    if 'Family  Size' in df.columns:
        df['Large_Family'] = (df['Family  Size'] > 4).astype(int)
    return df

train_data = add_features(train_data)
test_data = add_features(test_data)

# Fill missing values
for col in numerical_cols:
    if col in train_data.columns:
        median_value = train_data[col].median()
        train_data[col].fillna(median_value, inplace=True)
    if col in test_data.columns:
        median_value = test_data[col].median()
        test_data[col].fillna(median_value, inplace=True)

if 'Experience_Age_Ratio' in train_data.columns:
    train_data['Experience_Age_Ratio'].fillna(0, inplace=True)
if 'Experience_Age_Ratio' in test_data.columns:
    test_data['Experience_Age_Ratio'].fillna(0, inplace=True)

categorical_cols = ['Sex', 'Bachelor', 'Graduated', 'Career', 'Family Expenses', 'Variable', 'Age_Group']
encoder_dict = {}

for col in categorical_cols:
    if col in train_data.columns:
        encoder_dict[col] = LabelEncoder()
        train_data[col] = encoder_dict[col].fit_transform(train_data[col].astype(str))
    if col in test_data.columns:
        test_data[col] = encoder_dict[col].transform(test_data[col].astype(str))

# Feature and target variables
feature_cols = numerical_cols + [col for col in categorical_cols if col in train_data.columns] + ['Experience_Age_Ratio', 'Large_Family']
if 'Segmentation' not in train_data.columns:
    print("Error: 'Segmentation' column is missing in train data.")
    exit()
X = train_data[feature_cols]
y = train_data['Segmentation']

# Normalize numerical features
scaler = StandardScaler()
numerical_features = [col for col in numerical_cols + ['Experience_Age_Ratio'] if col in X.columns]
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Function to visualize numerical features
def plot_numerical_features(df, title, filename_prefix):
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if numerical_cols.empty:
        print(f"No numerical features found in {title}.")
        return
    for col in numerical_cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], kde=True, color="teal", bins=30)
        plt.title(f"Distribution of {col} in {title}", fontsize=16, weight='bold')
        plt.xlabel(col, fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"updated_visualizations/{filename_prefix}_numerical_{col}.png")
        plt.close()

# Visualize train and test numerical features
plot_numerical_features(train_data, "Train Dataset", "train_dataset")
plot_numerical_features(test_data, "Test Dataset", "test_dataset")

# Save a summary of the process
print("\nAll missing value details and visualizations saved in the 'updated_visualizations' and 'updated_reports' directories.")