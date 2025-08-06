from ydata_profiling import ProfileReport
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import warnings
from scipy import stats
from sklearn.preprocessing import QuantileTransformer
warnings.filterwarnings("ignore")


def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'The file path "{file_path}" was not found.')
    return pd.read_csv(file_path)

# Load the dataset
data = load_data(r'C:/Users/seglu/OneDrive/Desktop/Churn_prediction_model/EDA/copied_customer_churn_data.csv')

#Display the first few rows of the dataset
print(data.head())

# #Generate a profile report
# profile = ProfileReport(data, title="Customer Churn Data Profiling Report", explorative=True)
# # Save the report to an HTML file
# profile.to_file("C:/Users/seglu/OneDrive/Desktop/Churn_prediction_model/EDA/eda_report.html")
# #Display the report in a Jupyter Notebook (if applicable)
# profile.to_notebook_iframe()  

# Visualize the distribution of numerical features


def drop_corelated_features(df, cols_to_drop):
    drop_col = df.drop(cols_to_drop, axis=1)
    return drop_col

# Drop one column at a time and reassign
data = drop_corelated_features(data, 'avg_transaction_amount')
data = drop_corelated_features(data, 'transaction_count')
    

def separate_categorical_numerical(df):
    cat_col = []
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype == 'category':
            cat_col.append(col)
    return cat_col  # Corrected this line

# Call the function
categorical_col = separate_categorical_numerical(data)
print("Categorical Features:", categorical_col)


#univariance  Analysis

# def plot_categorical_distribution(df, features):
#     for feature in features:
#         plt.figure(figsize=(10, 6))
#         sns.countplot(data=df, x=feature, order=df[feature].value_counts().index)
#         plt.title(f'Distribution of {feature}')
#         plt.xlabel(feature)
#         plt.ylabel('Count')
#         plt.xticks(rotation=45)
#         plt.grid()
#         plt.show()

# # List of categorical features to visualize
# categorical_features = categorical_col
# plot_categorical_distribution(data, categorical_col)



def seperate_numerical_features(df):
    num_col = []
    for col in df.columns:
        if df[col].dtype == 'float64' or df[col].dtype == 'int64':
            num_col.append(col)

    return num_col  # Corrected this line  

# Call the function
numerical_col = seperate_numerical_features(data)
print("Numerical Features:", numerical_col)


# def plot_numerical_distribution(df, features):
#     for col in features:
#         plt.figure(figsize=(10, 6))
#         sns.histplot(df[col], kde=True, bins=50)
#         plt.title(f'Distribution of {col}')
#         plt.xlabel(col)
#         plt.xticks(rotation=45)
#         plt.ylabel('Frequency')
#         plt.grid()
#         plt.show()# List of numerical features to visualize


# plot_numerical_distribution(data, numerical_col)


# def plot_boxplots(df, features):
#     for feature in features:
#         plt.figure(figsize=(10, 6))
#         sns.boxplot(data=df, x=feature)
#         plt.title(f'Boxplot of {feature}')
#         plt.xlabel(feature)
#         plt.xticks(rotation=45)
#         plt.grid()
#         plt.show()

# # Call the function
# plot_boxplots(data, numerical_col)




# def mean_median_mode(df, features):
#     for feature in features:
#         mean = df[feature].mean()
#         median = df[feature].median()
#         mode = df[feature].mode()[0]  # mode() returns a Series, take the first value
#         print(f'Feature: {feature}')
#         print(f'Mean: {mean}, Median: {median}, Mode: {mode}\n')

# # Call the function
# mean_median_mode(data, numerical_col)


# def plot_boxplots_by_group(df, features, group_col):
#     for feature in features:
#         plt.figure(figsize=(10, 6))
#         sns.boxplot(data=df, x=group_col, y=feature)
#         plt.title(f'Boxplot of {feature} by {group_col}')
#         plt.xlabel(group_col)
#         plt.ylabel(feature)
#         plt.xticks(rotation=45)
#         plt.grid()
#         plt.tight_layout()
#         plt.show()

# plot_boxplots_by_group(data, numerical_col, group_col='customer_service_interaction_count')

# def plot_grouped_horizontal_plots(df, categorical_features, numeric_target='total_spent', group_col='age'):
#     for feature in categorical_features:
#         plt.figure(figsize=(12, 6))
#         sns.barplot(data=df, y=feature, x=numeric_target, hue=group_col)  # swapped x and y for horizontal plot
#         plt.title(f'{numeric_target} by {feature} grouped by {group_col}')
#         plt.xlabel(numeric_target)
#         plt.ylabel(feature)
#         plt.legend(title=group_col, bbox_to_anchor=(1.05, 1), loc='upper left')
#         plt.grid(axis='x')
#         plt.tight_layout()
#         plt.show()



# plot_grouped_horizontal_plots(data, categorical_col, numeric_target='total_spent', group_col='age')




# def plot_scatter_plots(df, features):
#     for feature in features:
#         plt.figure(figsize=(10, 6))
#         sns.scatterplot(data=df, x=feature, y='customer_service_interaction_count', alpha=0.6)
#         plt.title(f'Scatter Plot of {feature} vs total_spent')
#         plt.xlabel(feature)
#         plt.ylabel('Churn')
#         plt.grid()
#         plt.show()

# # Call the function
# plot_scatter_plots(data,categorical_col)

# def plot_boxplots_by_location(df, features):
#     for col in features:
#         plt.figure(figsize=(10, 6))
#         sns.scatterplot(data=df, x='login_frequency', y=col)
#         plt.title(f'{col} by login_frequency')
#         plt.xticks(rotation=45)
#         plt.grid()
#         plt.tight_layout()
#         plt.show()

# # Example call
# plot_boxplots_by_location(data, categorical_col)





# def standard_deviation_variance(df, features):
#     for feature in features:
#         std_dev = df[feature].std()
#         variance = df[feature].var()
#         print(f'Feature: {feature}')
#         print(f'Standard Deviation: {std_dev}, Variance: {variance}\n')

# # Call the function
# standard_deviation_variance(data, numerical_features)

# def correlation_matrix(df):
#     corr_matrix = df.corr()
#     plt.figure(figsize=(12, 8))
#     sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
#     plt.title('Correlation Matrix')
#     plt.show()

# # Note: Ensure that the numerical columns are correctly specified    

# correlation_matrix(data[numerical_col])



# def skewness_kurtosis(df, features):
#     for feature in features:
#         skewness = df[feature].skew()
#         kurtosis = df[feature].kurtosis()
#         print(f'Feature: {feature}')
#         print(f'Skewness: {skewness}, Kurtosis: {kurtosis}\n')

# # Call the function
# skewness_kurtosis(data, numerical_col)


1. #Function to compute skewness and kurtosis
def skewness_kurtosis(df, features):
    print("🔍 Skewness & Kurtosis")
    for feature in features:
        skewness = df[feature].skew()
        kurt = df[feature].kurtosis()
        print(f"Feature: {feature}")
        print(f"  Skewness: {skewness:.4f}, Kurtosis: {kurt:.4f}\n")

# 2. Function to perform Shapiro-Wilk normality test
def normality_test(df, features):
    print("📊 Normality Test (Shapiro-Wilk)")
    for feature in features:
        stat, p_value = stats.shapiro(df[feature])
        print(f"Feature: {feature}")
        print(f"  Statistic: {stat:.4f}, p-value: {p_value:.4f}")
        if p_value > 0.05:
            print(f"  ✅ {feature} is likely normally distributed.\n")
        else:
            print(f"  ❌ {feature} is NOT normally distributed.\n")

# Step 1: Analyze original data
skewness_kurtosis(data, numerical_col)
normality_test(data, numerical_col)


for feature in numerical_col:
    if (data[feature] < 0).any():
        print(f"❌ {feature} has negative values. Log1p not safe.")
    else:
        print('has no nagetive values')



numerical_col = [col for col in numerical_col if col != 'churn']


# Apply Quantile Transformer
qt = QuantileTransformer(output_distribution='normal', random_state=0)
data_qt_array = qt.fit_transform(data[numerical_col])

# Convert back to DataFrame for easier handling
data_qt = pd.DataFrame(data_qt_array, columns=numerical_col)

# Plot transformed distributions
for feature in numerical_col:
    plt.figure(figsize=(6, 4))
    sns.histplot(data_qt[feature], kde=True, bins=30)
    plt.title(f'Quantile Normal Transformed: {feature}')
    plt.tight_layout()
    plt.show()





# Select only the categorical columns from the original data
categorical_data = data[categorical_col]

# Concatenate with the transformed numerical DataFrame
data = pd.concat([categorical_data, data_qt,data['churn']], axis=1)

# Print or use the new combined dataset
print(data)
print(data.columns)


def copy_data(df):
    return df.copy()
# Call the function to copy the DataFrame
data_copy1 = copy_data(data)




def save_copy(df,file_path):
    df.to_csv(file_path, index =False)
    print(f'copied data and save :{file_path}')
save_copy(data_copy1, 'C:/Users/seglu/OneDrive/Desktop/Churn_prediction_model/EDA/customer_churn_data_copy1.csv')
    











































































































































































#Understand distributions of each column.
# for col in ['gender', 'location', 'account_type']:
#     sns.countplot(data=data, x=col)
#     plt.title(f'Count of {col}')
#     plt.xticks(rotation=45)
#     plt.show()


#Understand distributions of each column.
# for col in ['age', 'login_frequency', 'total_spent', 'customer_service_interaction_count']:
#     sns.histplot(data[col], kde=True)
#     plt.title(f'Distribution of {col}')
#     plt.show()


# #Check class balance:
# sns.countplot(x='churn', data=data)
# plt.title("Churn Distribution")
# plt.show()


# #Check how churn varies across categories.
# for col in ['gender', 'location', 'account_type']:
#     sns.countplot(data=data, x=col, hue='churn')
#     plt.title(f'{col} vs Churn')
#     plt.xticks(rotation=45)
#     plt.show()


# #See if churned users have different averages.
# for col in ['age', 'login_frequency', 'total_spent', 'customer_service_interaction_count']:
#     sns.boxplot(data=data, x='churn', y=col)
#     plt.title(f'{col} by Churn')
#     plt.show()

# #See if registration time affects churn.
# sns.boxplot(data=data, x='registration_date_month', y='churn')
# plt.title("Churn by Registration Month")
# plt.show()




# # Group by location and get average of numerical features
# location_stats = data.groupby('location')[['age', 'total_spent', 'churn']].mean().reset_index()

# # Plot average total spent by location
# sns.barplot(data=location_stats, x='location', y='total_spent')
# plt.title('Average Total Spent by Location')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()






#Make sure all values in the skew colums and they but be non-negative before applying:
# assert (data["total_spent"] >= 0).all(), "Negative values detected!"


# appling this bello





# ✅ Recommendation:
# Use Yeo-Johnson first — it's most reliable for general use.

# If you must achieve perfect normality (e.g., for a specific statistical model), then try QuantileTransformer.

# Don’t over-transform: if your model doesn’t require normality, just scale/standardize and move on.

# Would you like a function that automatically:

# Chooses between log1p, Yeo-Johnson, or QuantileTransformer?

# Applies the best one and plots before/after?





# from sklearn.preprocessing import PowerTransformer

# pt = PowerTransformer(method='yeo-johnson', standardize=True)
# transformed_data = pt.fit_transform(data[numerical_col])

# # Replace original columns or create new ones
# data_yeojohnson = data.copy()
# data_yeojohnson[numerical_col] = transformed_data



# for feature in skewed_features:
#     df[feature] = np.log1p(df[feature])
