import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler,OneHotEncoder,LabelEncoder,OrdinalEncoder
from sklearn.preprocessing import PowerTransformer
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os


def load_dataset(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'this {file_path} does not esist')

    return pd.read_csv(file_path)

data = load_dataset(r'C:\Users\seglu\OneDrive\Desktop\Churn_prediction_model\EDA\customer_churn_data_copy1.csv')

print(data.head())


#am trying to keep the extreme value in my total_spent column by using winsorize
#This keeps all data points but limits extreme outliers
def winsorize_series(df, lower_quantile=0.01, upper_quantile=0.99):
    lower = df.quantile(lower_quantile)
    upper = df.quantile(upper_quantile)
    return df.clip(lower, upper)


data['total_spent']=winsorize_series(data['total_spent'])

# using percentage value to normalise my columns here
def percentage_normalisation(df, col):
    if col not in df.columns:
        raise ValueError(f"The column '{col}' was not found in the DataFrame.")
    
    percentage_norm = df[col].value_counts(normalize=True)
    df[col] = df[col].map(percentage_norm)
    return df

#call the funtion here
data = percentage_normalisation(data, 'location')


def one_hot_encode(df, col):
    if col not in df.columns:
        raise ValueError(f"The column '{col}' was not found in the DataFrame.")
    
    encoder = OneHotEncoder(sparse_output=False, drop=None)  # use sparse_output
    encoded_array = encoder.fit_transform(df[[col]])
    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out([col]), index=df.index)
    
    df = df.drop(columns=[col])
    df = pd.concat([df, encoded_df], axis=1)
    
    return df

data = one_hot_encode(data, 'gender')



def ordinal_column_encode(df, col):
    # Optional: restrict to actual columns (object or category)
    if col not in df.select_dtypes(include=['object', 'category']).columns:
        raise ValueError(f"The column '{col}' must be of object or category dtype for encoding.")

    categories = [['savings', 'loan', 'checking']]

    encoder = OrdinalEncoder(categories=categories, handle_unknown='use_encoded_value', unknown_value=-1)
    df[col] = encoder.fit_transform(df[[col]])

    return df

# Example usage:
data = ordinal_column_encode(data, 'account_type')





def scaling_column(df, col):
    if col not in df.select_dtypes(include='number').columns:
        raise ValueError(f"The column '{col}' must be numerical.")
    
    scaler = MinMaxScaler()
    df[col] = scaler.fit_transform(df[[col]])

    return df

scaling_column(data, 'age')
scaling_column(data, 'login_frequency')
scaling_column(data,'registration_date_month')
scaling_column(data,'registration_date_day')


def save_copy(df,file_path):

    df.to_csv(file_path,index=False)

# save_copy(data, 'C:\Users\seglu\OneDrive\Desktop\Churn_prediction_model\EDA\churn_data_copy.csv')


save_copy(data, 'C:/Users/seglu/OneDrive/Desktop/Churn_prediction_model/data_copied.csv')








            
    
print(data.columns)


print(data.head())







# import numpy as np
# df['feature'] = np.log1p(df['feature'])  # log(1 + x) to handle zeros






















































































































































# data.drop('transaction_count', axis=1, inplace=True)
# data.drop('avg_transaction_amount', axis=1, inplace=True)





# def handle_skewness(df, threshold=0.5, method='yeo-johnson', target_col=None):
#     """
#     Detects and corrects skewness in numeric columns of a DataFrame, excluding the target column.

#     Parameters:
#     - df (pd.DataFrame): Input DataFrame.
#     - threshold (float): Skewness threshold above which transformation is applied.
#     - method (str): 'yeo-johnson' (default, supports 0 and negatives) or 'box-cox' (only for positive values).
#     - target_col (str): Optional column name to exclude from transformation.

#     Returns:
#     - df (pd.DataFrame): DataFrame with transformed skewed columns.
#     """
#     numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
#     if target_col and target_col in numeric_cols:
#         numeric_cols.remove(target_col)

#     skewed_cols = df[numeric_cols].apply(lambda x: x.skew()).sort_values(ascending=False)
#     skewed_cols = skewed_cols[abs(skewed_cols) > threshold].index.tolist()

#     if skewed_cols:
#         transformer = PowerTransformer(method=method)
#         df[skewed_cols] = transformer.fit_transform(df[skewed_cols])
    
#     return df

# handle_skewness(data,target_col='churn')
