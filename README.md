# MushroomClassification
# Mushroom Classification Project  This project focuses on building a machine learning model to classify mushrooms as **edible** or **poisonous** 
# Mushroom Classification Project

This project focuses on building a machine learning model to classify mushrooms as **edible** or **poisonous** based on their various physical characteristics. The dataset used contains multiple categorical features, such as cap shape, odor, and color, which are preprocessed and then used to train different machine learning models.

## Table of Contents

- [Dataset](#dataset)
- [Project Overview](#project-overview)
- [Data Preprocessing](#data-preprocessing)
- [Modeling](#modeling)
- [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
- [Evaluation](#evaluation)

## Dataset

The dataset used in this project is the **Mushroom Classification** dataset, which is publicly available on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Mushroom). It contains 8124 samples of mushrooms, each described by 22 categorical attributes.

## Project Overview

The primary goal of this project is to build a model that can accurately classify whether a mushroom is **edible** or **poisonous** based on its features. This involves:

1. **Data Cleaning**: Identifying and handling missing values and outliers.
2. **Data Encoding**: Applying label encoding and one-hot encoding to transform categorical data into a suitable format for machine learning models.
3. **Modeling**: Training and evaluating different machine learning models, including SVM, Naive  Bayes, Random Forest, Decision Tree, KNN, Logistic Regression, MLP.
4. **Evaluation**: Using metrics such as accuracy, confusion matrix, precision, recall, and F1-score to evaluate the performance of the models.

## Data Preprocessing

- **Label Encoding**: Encoded the target variable (`class`) using `LabelEncoder` to convert it into a numerical format.
- **One-Hot Encoding**: Applied `OneHotEncoder` to transform categorical features into binary vectors.

## Modeling

The following machine learning models were explored:

- **Support Vector Machine (SVM)**: Used to find the optimal hyperplane that separates edible and poisonous mushrooms with maximum margin.
- **Naive Bayes (MultinomialNB & GaussianNB)**: Applied to classify mushrooms based on the probability of features given the class.
- **Random Forest Classifier**: An ensemble method that trains multiple decision trees and combines their outputs for better generalization.
- **Decision Tree Classifier**: Built a decision tree to understand feature importance and model the decision boundaries.
- **K-Nearest Neighbors (KNN)**: Used to classify mushrooms based on the closest training examples in the feature space.
- **Logistic Regression**: A linear model used to predict the probability of mushrooms being edible or poisonous.
- **Multi-Layer Perceptron (MLP)**: Implemented a neural network to improve classification performance.
## Principal Component Analysis (PCA)

**Principal Component Analysis (PCA)** was used to reduce the dimensionality of the one-hot encoded data to 2 components, allowing for visualization in a 2D space. This helps in understanding the separation of data points between edible and poisonous mushrooms.

### PCA Scatter Plot

Below is the PCA scatter plot showing the two principal components:
![image](https://github.com/user-attachments/assets/c732f04c-4faa-45db-b811-c25630ab5a8e)

## Evaluation

Model performance was evaluated using various metrics:

- **Confusion Matrix**: Visualized the classification results.
- **Accuracy**: Calculated the overall correctness of the model.
- **Precision, Recall, and F1-Score**: Assessed the balance between precision and recall.

