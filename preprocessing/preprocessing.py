import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_and_clean_data(filepath):
    """
    Load and clean the dataset.
    """
    df = pd.read_csv(filepath)
    df['Chronic_Health_Conditions'] = df['Chronic_Health_Conditions'].fillna('No Disease')
    df['Dosage in mg'] = df['Dosage in mg'].fillna(0)
    df['Prescription'] = df['Prescription'].fillna('None')

    return df


def preprocess_health_conditions(df):
    """
    Create derived features based on chronic health conditions.
    """
    if 'Chronic_Health_Conditions_Diabetes' in df.columns:
        df['Age_Diabetes'] = df['Age'] * df['Chronic_Health_Conditions_Diabetes']
    if 'Chronic_Health_Conditions_Hypertension' in df.columns:
        df['Age_Hypertension'] = df['Age'] * df['Chronic_Health_Conditions_Hypertension']
    if 'Chronic_Health_Conditions_Heart Disease' in df.columns:
        df['Age_HeartDisease'] = df['Age'] * df['Chronic_Health_Conditions_Heart Disease']

    df['Age_Binned'] = pd.qcut(df['Age'], q=4, labels=False)
    df['Oxygen_HeartRate_Ratio'] = df['BloodOxygenLevel'] / df['HeartRate']

    return df


def tidy_data(df, regression_classification='classification'):
    """
    Map categorical variables to numerical values and create dummy variables.
    """
    mappings = {
        'Education_Level': {
            'No School': 0, 'Primary School': 1, 'Secondary School': 2, 'Diploma/Degree': 3
        },
        'Dominant_Hand': {
            'Left': 0, 'Right': 1
        },
        'Gender': {
            'Female': 0, 'Male': 1
        },
        'Family_History': {
            'No': 0, 'Yes': 1
        },
        'Smoking_Status': {
            'Current Smoker': 0, 'Former Smoker': 1, 'Never Smoked': 2
        },
        'APOE_ε4': {
            'Negative': 0, 'Positive': 1
        },
        'Depression_Status': {
            'No': 0, 'Yes': 1
        },
        'Medication_History': {
            'No': 0, 'Yes': 1
        },
        'Sleep_Quality': {
            'Poor': 0, 'Good': 1
        }
    }

    for column, mapping in mappings.items():
        if column in df.columns:
            df[column] = df[column].map(mapping)

    nominal_columns = ['Chronic_Health_Conditions', 'Physical_Activity', 'Nutrition_Diet']
    if regression_classification == 'regression':
        nominal_columns.append('Prescription')

    df = pd.get_dummies(df, columns=nominal_columns, drop_first=False)  # drop_first=False keeps all categories
    df = df.apply(lambda x: x.astype(int) if x.dtype == bool else x)
    preprocess_health_conditions(df)
    return df


def normalize_data(df, regression_classification='classification'):
    """
    Normalize numerical columns using StandardScaler.
    """
    columns_to_include = [
        'HeartRate', 'BloodOxygenLevel', 'BodyTemperature', 'Weight', 'MRI_Delay',
        'Age', 'Oxygen_HeartRate_Ratio'
    ]

    if regression_classification == 'classification':
        df = df.drop(columns=['Dosage in mg', 'Prescription'])

    numerical_cols = [col for col in df.columns if col in columns_to_include]
    scaler = StandardScaler()

    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df
