import pandas as pd            # For data manipulation
import os                      # For file/path operations
import warnings                # To suppress warnings
from sklearn.utils import resample
#  from imblearn.over_sampling import SMOTEEN # For oversampling imbalanced classes
# from imblearn.over_sampling import ADASYN
warnings.filterwarnings('ignore')         # Hides warning messages (useful during cleaning/modeling)
from sklearn.model_selection import train_test_split




def load_dataset(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'The file "{file_path}" does not exist or was not found.')
    
    return pd.read_csv(file_path)

# Load your dataset
data = load_dataset(r'C:\Users\seglu\OneDrive\Desktop\Churn_prediction_model\data_copied.csv')

print(data.head(5))
print(data.columns)


# Move 'churn' to the end
target = 'churn'
columns = [col for col in data.columns if col != target] + [target]
data = data[columns]





def balance_data_by_upsampling(df, target_column='target', random_state=42):
    """
    Balances binary classification data using random oversampling of the minority class.

    Parameters:
    - df: pandas DataFrame, including both features and target
    - target_column: name of the target column
    - random_state: seed for reproducibility

    Returns:
    - X: balanced feature DataFrame
    - y: balanced target Series
    """
    # Separate classes
    majority_class = df[target_column].value_counts().idxmax()
    minority_class = df[target_column].value_counts().idxmin()

    majority = df[df[target_column] == majority_class]
    minority = df[df[target_column] == minority_class]

    # Upsample minority
    minority_upsampled = resample(minority,
                                   replace=True,
                                   n_samples=len(majority),
                                   random_state=random_state)

    # Combine and shuffle
    df_balanced = pd.concat([majority, minority_upsampled]).sample(frac=1, random_state=random_state)

    X = df_balanced.drop(columns=[target_column])
    y = df_balanced[target_column]

    return X, y

# Assuming your original data is in a DataFrame called df
X_balanced, y_balanced = balance_data_by_upsampling(data, target_column='churn')

print("Balanced class distribution:")
print(y_balanced.value_counts())



def training_dataset(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = training_dataset(X_balanced, y_balanced)

print("Train shape:", X_train.shape, y_train.shape)
print("Test shape:", X_test.shape, y_test.shape)




def save_train_test_data(X_train, X_test, y_train, y_test, folder='data'):
    """
    Save train and test datasets to CSV files inside the specified folder.
    Creates the folder if it doesn't exist.

    Parameters:
    - X_train, X_test: pandas DataFrames of features
    - y_train, y_test: pandas Series or DataFrames of targets
    - folder: folder path to save the files (default 'data')
    """
    os.makedirs(folder, exist_ok=True)

    X_train.to_csv(os.path.join(folder, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(folder, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(folder, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(folder, 'y_test.csv'), index=False)


# After splitting your data
save_train_test_data(X_train, X_test, y_train, y_test, folder='saved_data')

print(X_train.columns)


