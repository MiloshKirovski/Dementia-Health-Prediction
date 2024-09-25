import joblib
import numpy as np
import matplotlib.pyplot as plt
from preprocessing.preprocessing import load_and_clean_data, tidy_data, normalize_data
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def split_data_classification(df, target_column):
    """
    Split the data into training and testing sets for classification.
    """
    X = df.drop(columns=[
        target_column,
        'Cognitive_Test_Scores',
        'Nutrition_Diet_Mediterranean Diet',
        'Chronic_Health_Conditions_Diabetes',
        'Diabetic',
        'Chronic_Health_Conditions_No Disease',
        'Nutrition_Diet_Balanced Diet'
    ])
    y = df[target_column]

    return train_test_split(X, y, test_size=0.2, random_state=42)


def grid_search_random_forest_classification(X_train, y_train):
    """
    Perform a grid search to find the best hyperparameters for RandomForest.
    """
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    print("Best parameters found for classification:", grid_search.best_params_)

    return grid_search.best_estimator_


def evaluate_classification_model(model, X_test, y_test):
    """
    Evaluate the classification model and print accuracy.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Classification Model Accuracy: {accuracy:.2f}\n')

    report = classification_report(y_test, y_pred, target_names=['No Dementia', 'Dementia'])
    print('Classification Report:')
    print(report)

    return accuracy


def plot_feature_importance(model, features):
    """
    Plot feature importance of the model.
    """
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)[::-1]
    sorted_features = features[sorted_idx]
    sorted_importance = feature_importance[sorted_idx]

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features, sorted_importance)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance in Random Forest Classifier')
    plt.gca().invert_yaxis()
    plt.show()


def save_model(model, filename):
    """
    Save the model to a file.
    """
    joblib.dump(model, filename)
    print(f'Model saved to {filename}')


def load_model(filename):
    """
    Load the model from a file.
    """
    model = joblib.load(filename)
    print(f'Model loaded from {filename}')
    return model


def main(filepath):
    """
    Main function for model training.
    """
    df = load_and_clean_data(filepath)
    df = tidy_data(df)
    df = normalize_data(df)

    # Split data for classification
    X_train_class, X_test_class, y_train_class, y_test_class = split_data_classification(df, 'Dementia')

    model_filename = 'random_forest_classifier.pkl'
    try:
        rf_model_class = load_model(model_filename)
    except FileNotFoundError:
        rf_model_class = grid_search_random_forest_classification(X_train_class, y_train_class)
        save_model(rf_model_class, model_filename)

    print("Random Forest Classification Model Performance:")
    evaluate_classification_model(rf_model_class, X_test_class, y_test_class)
    plot_feature_importance(rf_model_class, X_train_class.columns)


if __name__ == "__main__":
    filepath = '../data/dementia_patients_health_data.csv'
    main(filepath)
