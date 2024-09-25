import pandas as pd
from preprocessing.preprocessing import load_and_clean_data
from visualization.feature_visualization import visualize_data


def display_data_summary(df):
    """
    Display a summary of the dataframe: unique values, description, and missing values.
    """
    pd.set_option('display.max_columns', None)

    print(df.head(), end='\n\n')
    print(f'Data shape: {df.shape}\n')
    print(f'Data types:\n{df.dtypes}\n')
    print(f'Columns:\n{df.columns.tolist()}\n')

    for col in df.columns:
        unique_values = df[col].unique()
        if len(unique_values) > 10:
            print(f'Column {col} - Unique values (showing first 5 and last 5):')
            print(f'{list(unique_values)[:5]} ... {list(unique_values)[-5:]}')
        else:
            print(f'Column {col} - Unique values:')
            print(unique_values)
        print()

    print('Data description:')
    print(df.describe(), end='\n\n')

    print('Missing values per column:')
    print(df.isnull().sum())


def main():
    """
    Main entry point for the script.
    """
    pd.set_option('display.max_columns', None)
    filepath = '../data/dementia_patients_health_data.csv'

    try:
        df = load_and_clean_data(filepath)
        display_data_summary(df)

        visualize_data(df)
    except FileNotFoundError:
        print(f'Error: The file at {filepath} was not found.')
    except Exception as e:
        print(f'An error occurred: {e}')


if __name__ == '__main__':
    main()
