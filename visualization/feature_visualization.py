import os
import seaborn as sns
import matplotlib.pyplot as plt
from preprocessing.preprocessing import load_and_clean_data


def visualize_data(df):
    sns.set(style='whitegrid')

    # Create a directory to save the plots
    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)

    # 1. Distribution of Chronic Health Conditions
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x='Chronic_Health_Conditions', hue='Dementia', palette="Set2")
    plt.title('Distribution of Chronic Health Conditions by Dementia Status')
    plt.savefig(os.path.join(output_dir, 'chronic_health_conditions_distribution.png'))
    plt.close()

    # 2. Bar plot for Education Level
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x='Education_Level', hue='Dementia', palette='Paired')
    plt.title('Education Level Distribution')
    plt.savefig(os.path.join(output_dir, 'education_level_distribution.png'))
    plt.close()

    # 3. Correlation heatmap
    plt.figure(figsize=(12, 8))
    numerical_df = df.select_dtypes(include=['float64', 'int64'])
    corr_matrix = numerical_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='viridis', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()

    # 4. Pairplot for selected numerical features
    pairplot = sns.pairplot(df, vars=['Age', 'Weight', 'Diabetic'], hue='Physical_Activity', palette='husl')
    pairplot.fig.suptitle('Pairplot of Age, Weight, and Physical Activity', y=1.02)
    pairplot.savefig(os.path.join(output_dir, 'pairplot_age_weight_physical_activity.png'))
    plt.close()

    # 5. Age - Dementia Distribution using Violin Plot
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df, x='Dementia', y='Age', hue='Gender', split=True, palette="Set2")
    plt.title('Violin Plot of Age Distribution by Gender and Dementia Status')
    plt.xlabel('Dementia Status')
    plt.ylabel('Age')
    plt.savefig(os.path.join(output_dir, 'age_distribution_violin_plot.png'))
    plt.close()

    # 6. Dementia - Cognitive Score Relationship
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Dementia', y='Cognitive_Test_Scores', hue='Gender', palette='Blues')
    plt.title('Cognitive Test Scores by Dementia Status')
    plt.xlabel('Dementia Status')
    plt.ylabel('Cognitive Test Scores')
    plt.savefig(os.path.join(output_dir, 'cognitive_scores_box_plot.png'))
    plt.close()

    # 7. Physical Activity vs. Dementia Status
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x='Physical_Activity', hue='Dementia', palette="Set1")
    plt.title('Physical Activity Distribution by Dementia Status')
    plt.xlabel('Physical Activity Level')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'physical_activity_distribution.png'))
    plt.close()

    # 8. Dosage Distribution for Dementia and Non-Dementia Patients
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Dementia', y='Dosage in mg', hue='Depression_Status', palette="Set1")
    plt.title('Dosage Distribution for Dementia Patients vs. Non-Dementia Patients')
    plt.xlabel('Dementia Status')
    plt.ylabel('Dosage in mg')
    plt.savefig(os.path.join(output_dir, 'dosage_distribution_box_plot.png'))
    plt.close()

