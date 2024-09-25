import joblib
import numpy as np
from preprocessing.preprocessing import (
    load_and_clean_data,
    preprocess_health_conditions,
    tidy_data,
    normalize_data
)
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE


def prepare_data(filepath):
    """
    Load, clean, preprocess, and normalize the data for regression.
    """
    df = load_and_clean_data(filepath)
    df = preprocess_health_conditions(df)
    df = tidy_data(df, regression_classification='regression')
    df = normalize_data(df, regression_classification='regression')
    return df


def split_data(df):
    """
    Split the data into features (X) and target (y).
    """
    X = df.drop(columns=['Cognitive_Test_Scores'], axis=1)
    y = df['Cognitive_Test_Scores']
    return train_test_split(X, y, test_size=0.2, random_state=42)


def select_features(X, y):
    """
    Apply feature selection methods and return selected features.
    """
    feature_sets = {}

    # Method 1: SelectKBest
    print("\nRunning SelectKBest:")
    for k in [10, 20, 30]:  # Try different values for k
        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        feature_sets[f'SelectKBest (k={k})'] = selected_features
        print(f"Top {k} features selected using SelectKBest: {selected_features}")

    # Method 2: Recursive Feature Elimination (RFE)
    print("\nRunning RFE:")
    rfe = RFE(GradientBoostingRegressor(random_state=42), n_features_to_select=20)
    rfe.fit(X, y)
    rfe_features = X.columns[rfe.get_support()].tolist()
    feature_sets['RFE'] = rfe_features
    print(f"Features selected using RFE: {rfe_features}")

    return feature_sets


def cross_validate(X, y, feature_sets):
    """
    Cross-validate the Gradient Boosting model with selected features and return the best RFE feature set.
    """
    model = GradientBoostingRegressor(random_state=42)
    best_r2 = float('-inf')
    best_features = None

    for method, features in feature_sets.items():
        print(f"\nCross-validating Gradient Boosting model with features from {method}:")
        X_subset = X[features]

        # Cross validation
        scores = cross_val_score(model, X_subset, y, cv=5, scoring='r2')
        print(f"R-squared scores: {scores}")
        mean_r2 = np.mean(scores)
        print(f"Mean R-squared: {mean_r2}")

        # Update the best RFE feature set if this one is better
        if method == 'RFE' and mean_r2 > best_r2:
            best_r2 = mean_r2
            best_features = features

    return best_features


def train_and_evaluate(X_train, X_test, y_train, y_test, selected_features):
    """
    Train the Gradient Boosting model and evaluate its performance.
    """
    gb_params = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.3],
        'max_depth': [3, 4, 5]
    }
    gb = GradientBoostingRegressor(random_state=42)
    gb_grid = GridSearchCV(gb, gb_params, cv=5, n_jobs=-1, verbose=1)
    gb_grid.fit(X_train[selected_features], y_train)

    # Model evaluation
    y_pred = gb_grid.predict(X_test[selected_features])
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\nGradient Boosting Model:")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"R-squared Score: {r2}")

    return gb_grid


def display_feature_importance(best_model, selected_features):
    """
    Display the top 10 most important features from the trained model.
    """
    feature_importance = best_model.feature_importances_
    feature_importance_dict = dict(zip(selected_features, feature_importance))
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

    print("\nTop 10 Most Important Features:")
    for feature, importance in sorted_features[:10]:
        print(f"{feature}: {importance}")


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


def main():
    filepath = "../data/dementia_patients_health_data.csv"
    df = prepare_data(filepath)

    model_filename = 'gradient_boosting_regressor.pkl'

    X, y = df.drop(columns=['Cognitive_Test_Scores'], axis=1), df['Cognitive_Test_Scores']
    X_train, X_test, y_train, y_test = split_data(df)

    try:
        gb_model = load_model(model_filename)
        selected_features = gb_model.feature_names_in_
    except FileNotFoundError:
        feature_sets = select_features(X, y)
        selected_features = cross_validate(X, y, feature_sets)

        gb_model = train_and_evaluate(X_train, X_test, y_train, y_test, selected_features)
        save_model(gb_model, model_filename)

    display_feature_importance(gb_model.best_estimator_, selected_features)


if __name__ == "__main__":
    main()
