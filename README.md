# Movie Rating Prediction using Machine Learning
Machine learning project to predict movie ratings (vote_average) based on metadata features from The Movies Dataset on Kaggle. This project implements a full ML pipeline from data acquisition to model evaluation, featuring exploratory data analysis, feature engineering and comparative model assessment.
## Overview
A data science project that builds a regression model to predict user ratings for movies based on their metadata (budget, revenue, genres, popularity, etc.). The system analyzes 45,466 movies, applies data cleaning and feature engineering and compares multiple machine learning algorithms to identify the optimal predictor.
The project follows a complete ML workflow:
-  **Data Acquisition & Cleaning**: Handles missing values, removes invalid entries and ensures data integrity
- **Exploratory Data Analysis (EDA)**: Visualizes distributions, correlations and patterns in the data
- **Feature Engineering**: Creates meaningful derived features (is_high_budget, main_genre)
- **Model Selection**: Compares Linear Regression, Decision Tree and Random Forest using 5-Fold Cross-Validation
- **Final Evaluation**: Tests the best model on a held-out test set with comprehensive metrics

## Key Features
### Data Processing Pipeline
- **Robust Cleaning**: Removes entries with invalid ratings, zero/negative budgets and revenues, and invalid release years
- **Smart Imputation**: Uses median imputation for missing numerical values and "Unknown" for missing categorical values
### Exploratory Data Analysis (EDA)
- **Distribution Analysis**: Visualizes rating distributions, budget/revenue correlations
- **Temporal Trends**: Analyzes rating evolution over time (release years)
- **Genre Insights**: Examines rating patterns across different movie genres
- **Correlation Matrix**: Identifies relationships between numerical features
### Feature Engineering
- `main_genre`: Extracts the primary genre from complex genre lists
- `is_high_budget`: Binary indicator for high-budget productions (above median)
### Model Selection & Validation
- **Multiple Algorithms**: Compares Linear Regression, Decision Tree (CART) and Random Forest
- **5-Fold Cross-Validation**: Ensures robust performance estimation on training data
- **Hold-out Test Set**: Final evaluation on unseen 15% of data
### Comprehensive Metrics
- **MAE (Mean Absolute Error)**: 0.494 (average prediction error in rating points)
- **MSE (Mean Squared Error)**: 0.453 (penalizes larger errors)
- **RMSE (Root Mean Squared Error)**: 0.673 (interpretability in rating points)
- **R² (Coefficient of Determination)**: 0.421 (explains 42% of variance)
- **MAPE (Mean Absolute Percentage Error)**: 8.68% (relative error)

## Tech Stack
- **Language**: Python
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-Learn (LinearRegression, DecisionTreeRegressor, RandomForestRegressor)

##  Dataset
The dataset used is the **The Movies Dataset**.
**Download:** [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?resource=download).
**Description:** 45,466 movies with 24 features.
