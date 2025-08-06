import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import os
warnings.filterwarnings("ignore")

print(os.getcwd())
# Set the working directory
# os.chdir('C:/Users/seglu/OneDrive/Desktop/Churn_prediction_model/DATA_CLEANING')




# Load the dataset
# data = pd.read_csv('customer_churn_data.csv')




# data = pd.read_csv('C:/Users/seglu/OneDrive/Desktop/Churn_prediction_model/DATA_CLEANING/customer_churn_data.csv')


# # Display the data
# print(data)

# # Display the first few rows of the dataset
# print(data.head())

# # Display the shape of the dataset
# print('shape of the dataset:', data.shape)

# # Display the data types of each column
# print('data types of each column :',data.dtypes)

# # Drop the 'customer_id' columm as it is not need for analysis
# data.drop('customer_id', axis =1, inplace=True)

# # Display the summary statistics of the dataset
# print('summary statistics of the dataset:', data.describe())

# #Display the column names
# print('column names :',data.columns)




# # Display the number of missing values in each column
# print('number of missing values in each column:', data.isnull().sum())      


# # Display the number of unique values in each column
# print('number of unique values in each column:', data.nunique())   

# # Display the column information
# print('column information:',data.info())


# # Display the row duplicates
# print('number of row duplicates:',data.duplicated().sum())


# # To treat the null values, we can wither drop the row or fill the null values with a specific value
# # Fill the null values with the mean of the column
# data.account_type.fillna(data.account_type.mode()[0], inplace= True)

# # Fill the null values with the mode of the column
# data.gender.fillna(data.gender.mode()[0], inplace = True)


# # convert the 'registration_date' column to datetime format
# # data['registration_date'] = pd.to_datetime(data['registration_date'],format = '%Y-%m-%d')



# # Display the summary statistics of the dataset after filling the null values
# print('summary statistics of the dataset after filling the null values:', data.describe())

# # Display the number of object column in the dataset
# num_object_columns = data.select_dtypes(include=['object']).shape[1]
# print('number of object column in the dataset:',num_object_columns)

# # Display the Object columns in the dataset
# object_columns = data.select_dtypes(include=['object']).columns.tolist()
# print('Object columns in the dataset:', object_columns)

# # Display the number of numeric columns in the dataset
# num_numeric_columns = data.select_dtypes(include = ['number']).shape[1]
# print('Number of numeric columns in the dataset:',num_numeric_columns)

# # Display the numerical columns in the dataset
# numeric_columns = data.select_dtypes(include = ['number']).columns.tolist()
# print('Numerical columns in the dataset:', numeric_columns)


# # Dis play the number of categorical columns in the dataset
# num_categorical_columns = data.select_dtypes(include=['category']).shape[1]
# print('Number of categorical columns in the dataset:', num_categorical_columns)




def load_data(file_path):

    print(f"Loading data from {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
  
    return pd.read_csv(file_path)
# Load the dataset
data =load_data('C:/Users/seglu/OneDrive/Desktop/Churn_prediction_model/DATA_CLEANING/customer_churn_data.csv')


def analyze_column_type(df):
    column_info = pd.DataFrame({
        'Column Name': df.columns,
        'Data Type': df.dtypes.values,
        'Missing Values': df.isnull().sum().values,
        'Unique Values': [df[col].nunique() for col in df.columns],
        'check number of row and column': df.shape[0],
        'check number of columns': df.shape[1],
        # 'check for shape': df.shape()

    })
    return column_info

# Then call it:
column_info = analyze_column_type(data)
print(column_info)


def drop_column(df, column_name):

    df_drop = df.drop(column_name,axis=1,inplace = True)

    if df_drop is not None:
        print(f"Column '{column_name}' has been dropped from the DataFrame.")
    else:           
        print(f"Column '{column_name}' does not exist in the DataFrame.")

    return df_drop
# Call the function to drop the 'customer_id' column
drop_column(data, 'customer_id')

    
    


def checking_for_duplicates(df):
    # creating a container to store my duplicate rows

    duplicate = df[df.duplicated()]

    if  duplicate.empty:
        print(f'their are {duplicate} in my datase')

    else:
        print(F'their are  {duplicate} duplicate in my dataset')

    return duplicate

# Call the function to check for duplicates
checking_for_duplicates(data)


def missing_values(df):
    # fill missing values with mode for categorical columns

    
    miss_value = pd.Series({
        "missing values for account_type": df.account_type.fillna(df.account_type.mode()[0], inplace = True),
        "missing values for gender": df.gender.fillna(df.gender.mode()[0], inplace = True)

    }) 

    # df.account_type.fillna(df.account_types.mode()[0], inplace = True) 
    
    return miss_value

# Call the function to get missing values
missing_values(data)


def checking_the_null_after_filling(df):
    

    null_values = df.isnull().sum()


    if null_values.sum() == 0:
        print("There are no null values in the dataset after filling missing values.")
    else:
        print("There are still null values in the dataset after filling missing values:")
        print(null_values[null_values > 0])


    return null_values

# Call the function to check for null values after filling
checking_the_null_after_filling(data)
    
    


def statistics_summary(df):
    #summary statistics for numeric columns
    numeric_summary = df.describe()

    #statistics for categorical columns
    categorical_summary = df.describe(include = ['object','category'])

    print("Statistics for numerical columns:")
    print(' -' * 40)

    print('Statistics for categorical columns:')
    print(' -' * 40)

    return numeric_summary, categorical_summary

# Call the function to get the statistics summary
statistics_summary(data) 





def analyze_column_types(df):
    # Number and names of object (string) columns
    num_object_columns = df.select_dtypes(include=['object']).shape[1]
    object_columns = df.select_dtypes(include=['object']).columns.tolist()

    # Number and names of numeric columns
    num_numeric_columns = df.select_dtypes(include=['number']).shape[1]
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

    # Number and names of categorical columns
    num_categorical_columns = df.select_dtypes(include=['category']).shape[1]
    categorical_columns = df.select_dtypes(include=['category']).columns.tolist()

    # Print summary
    print("Dataset Column Type Analysis")
    print(f"Number of object columns: {num_object_columns}")
    print(f"Object columns: {object_columns}")
    print(f"Number of numeric columns: {num_numeric_columns}")
    print(f"Numeric columns: {numeric_columns}")
    print(f"Number of categorical columns: {num_categorical_columns}")
    print(f"Categorical columns: {categorical_columns}")

    # Return results as a dictionary if needed
    return {
        'object': object_columns,
        'numeric': numeric_columns,
        'categorical': categorical_columns
    }
# Call the function to analyze column types
column_types = analyze_column_types(data)


def extract_column_info(df, column_name):
    # Convert to datetime if not already
    df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
    
    # Extract year, month, day
    df[f'{column_name}_year'] = df[column_name].dt.year
    df[f'{column_name}_month'] = df[column_name].dt.month
    df[f'{column_name}_day'] = df[column_name].dt.day

    return df

# Then call it
data = extract_column_info(data, 'registration_date')
drop_column(data, 'registration_date')  # Drop the original column if needed
# Display the updated DataFrame                     
print('data after extracting column info:', data.head())



print('checking null value again',data.isnull().sum().sort_values(ascending=False))


def check_for_null_values(df):
    # Check for null values in each column and print the results
    
    for col in df.columns:
        if df[col].value_counts().sum():
            print(f"Column '{col}' has no values.")

    
        else:
            print(f"Column '{col}' has no null values.")    

# Call the function to check for null values
check_for_null_values(data)





def check_for_balance_class(df, target_column):
    """
    Check the balance of classes in the target column.
    
    Parameters:
    df (DataFrame): The DataFrame containing the data.
    target_column (str): The name of the target column to check.
    
    Returns:
    None
    """
    class_counts = df[target_column].value_counts()
    print(f"Class distribution in '{target_column}':")
    print(class_counts)
    
    # Plotting the class distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(x=target_column, data=df, palette='viridis')
    plt.title(f'Class Distribution of {target_column}')
    plt.xlabel(target_column)
    plt.ylabel('Count')
    plt.show()

# Call the function to check for balance in the 'churn' column
check_for_balance_class(data, 'churn')
check_for_balance_class(data,'gender')





def find_numeric_strings(df):
    # Find columns that are of object type but can be converted to numeric
    numeric_strings = []
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                pd.to_numeric(df[col])
                numeric_strings.append(col)
            except ValueError:
                pass
    return numeric_strings



# Call the function to find numeric strings
num=find_numeric_strings(data)
print('are you sure none are object',num)


def clean_strings(df):
    """
    Clean string columns in the DataFrame by removing leading/trailing spaces and converting to lowercase.
    
    Parameters:
    df (DataFrame): The DataFrame containing the data.
    
    Returns:
    DataFrame: The cleaned DataFrame.
    """
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip().str.lower()
    return df

# Call the function to clean strings
data = clean_strings(data)


def copy_data(df):
    return df.copy()
# Call the function to copy the DataFrame
data_copy = copy_data(data)

def save_copy_data(df, file_path):
    """
    Save the copied DataFrame to a CSV file.
    
    Parameters:
    df (DataFrame): The DataFrame to save.
    file_path (str): The path where the CSV file will be saved.
    """
    df.to_csv(file_path, index=False)
    print(f"Copied data saved to {file_path}")

# Save the copied data to a new CSV file
save_copy_data(data_copy, 'C:/Users/seglu/OneDrive/Desktop/Churn_prediction_model/DATA_CLEANING/copied_customer_churn_data.csv')



print('shape of the data:', data.shape)


















