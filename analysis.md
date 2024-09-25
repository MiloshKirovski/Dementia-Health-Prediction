# Analysis of Dementia Patient Data: Classification and Regression Models

This analysis focuses on modeling dementia patient data to classify individuals and predict outcomes using machine learning techniques. Two models are employed: one for dementia classification and another regression model that predicts cognitive scores based on all other variables. Since predicting dementia using all variables yields an accuracy of nearly 100%, I decided to exclude features such as prescription information, dosage in milligrams, and cognitive test scores that could provide direct answers. Instead, I concentrated on the remaining variables to develop a more nuanced estimation for dementia classification. Notably, the model identified the Depression_Status variable as the most significant predictor, which serves as a valuable starting point for understanding the model I constructed.

## 

## Data Overview

- **Number of Records**: 1000

- **Features**: This dataset includes a range of demographic, clinical, and lifestyle factors:
  
  - **Demographic**: Age, Education_Level, Gender, Dominant_Hand, Family_History, Smoking_Status, APOE_ε4
  - **Health Metrics**: HeartRate, BloodOxygenLevel, BodyTemperature, Weight, MRI_Delay
  - **Medical History**: Diabetic, Depression_Status, Cognitive_Test_Scores, Chronic_Health_Conditions (including conditions like heart disease, diabetes, hypertension)
  - **Prescriptions**: Dosage in mg, Prescription (Donepezil, Galantamine, Memantine, None)
  - **Lifestyle Factors**: AlcoholLevel, Physical_Activity, Nutrition_Diet, Sleep_Quality, Medication_History

- **Feature Types**: The features consist of both categorical (e.g., Gender, Smoking_Status) and continuous variables (e.g., Age, HeartRate).

## 

## Classification Model Analysis

### Model Overview

I developed a Random Forest classification model to predict the likelihood of dementia based on various health and lifestyle features. I chose Random Forest because of its ability to handle high-dimensional data and its interpretability, particularly through feature importance analysis. To prevent overly straightforward predictions, I excluded certain features, such as **Cognitive Test Scores** and specific nutrition and health conditions, that would have made the dementia classification too obvious.

### Hyperparameter Tuning

Best parameters:

- `max_depth`: None
- `min_samples_leaf`: 2
- `min_samples_split`: 5
- `n_estimators`: 100

These parameters balance the complexity of the model without overfitting.

### Model Performance

The model achieved an **accuracy of 78%** on the test set. While accuracy provides an initial assessment of the model’s performance, I also evaluated **precision** and **recall** to ensure it wasn’t biased in classifying dementia cases. The precision for predicting non-dementia was **73%**, and for dementia, it was **84%**. The recall for non-dementia was **84%**, and for dementia, it was **73%**, resulting in an overall **F1 score** of 78% for both classes of the model.

In addition, I generated a feature importance plot, which showed that **Depression Status** was the most significant predictor, followed by **APOE_ε4 **and then **Smoking_Status**. This finding aligns with existing research, suggesting a strong correlation between mental health and dementia risk.

I saved the model in a `.pkl` file for future reference and further analysis.



## Regression Model Analysis

### Model Overview

I built a Gradient Boosting regression model to predict **Cognitive Test Scores** based on a range of health and lifestyle factors. Gradient Boosting was my choice because of its robustness in handling non-linear relationships and feature interactions, which seemed appropriate for the data. My primary goal was to identify which features most significantly contributed to variations in cognitive test scores and to build a reliable predictive model.

### Data Preparation and Feature Selection

I began by cleaning and normalizing the data, focusing on health conditions and other relevant factors. I excluded **Cognitive Test Scores** from the feature set to avoid any direct leakage into the target variable.

For feature selection, I explored two methods: **SelectKBest** and **Recursive Feature Elimination (RFE)**.

- **SelectKBest:** I tested the model with different numbers of top features (10, 20, and 30) to see how the regression model performed. The top 10 features included important predictors like **Dosage in mg, Smoking Status**, and **APOE_ε4**.

- **RFE (Recursive Feature Elimination):** This method identified 20 important features, including **AlcoholLevel, HeartRate, BloodOxygenLevel**, and **Depression_Status**. After cross-validation, I found that the RFE-selected features outperformed the SelectKBest features.

### Cross-Validation Results

I cross-validated the Gradient Boosting model using both SelectKBest and RFE-selected features. The RFE feature set performed the best, with an average **R-squared of 0.695**. This indicated a strong predictive capability, especially compared to SelectKBest with fewer features. 

### Model Performance

After tuning the hyperparameters using GridSearchCV, I evaluated the model on the test set. The final model achieved an **R-squared of 0.686** and a **Root Mean Squared Error (RMSE) of 1.83**, which suggested that the model could explain nearly 70% of the variance in cognitive test scores. While not perfect, it provided reasonable accuracy in predicting cognitive decline based on the selected features.

### Feature Importance

The feature importance analysis revealed some interesting insights:

- **Dosage in mg** was the most influential feature, contributing about **38%** to the prediction. This result aligns with expectations, given the potential effects of medications on cognitive performance.
- **Prescription_None** and **Dementia** were also significant contributors, reflecting the impact of medication and the presence of dementia on cognitive test scores.
- Other notable features included **MRI_Delay**, **BodyTemperature**, and **AlcoholLevel**, though their contributions were smaller.


