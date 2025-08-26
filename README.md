# Store Sales Time Series Forecasting 🏪📈

A comprehensive machine learning pipeline for predicting store sales using time series data from the Kaggle Store Sales Time Series Forecasting competition.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Key Components](#key-components)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements an end-to-end machine learning pipeline for forecasting store sales using historical sales data, store information, holidays, oil prices, and transaction data. The solution includes comprehensive data preprocessing, feature engineering, multiple model training, and automated evaluation.

### 🏆 Competition Details
- **Competition**: [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
- **Objective**: Predict sales for different product families across multiple stores
- **Evaluation Metric**: Root Mean Squared Logarithmic Error (RMSLE)

## ✨ Features

### 🔧 Data Preprocessing
- **Multi-source data integration** (train, test, holidays, oil prices, transactions, store info)
- **Advanced date feature engineering** with cyclical encoding
- **Holiday processing** (national, regional, local)
- **Oil price trend analysis** with moving averages
- **Transaction pattern analysis**
- **Robust missing value handling**

### 🤖 Machine Learning Models
- **Linear Models**: LinearRegression, Ridge, ElasticNet, SGDRegressor
- **Ensemble Methods**: RandomForest, ExtraTrees, GradientBoosting, Bagging
- **Meta-Ensemble**: VotingRegressor for combining best models
- **Time Series Cross-Validation** for proper model validation

### 📊 Evaluation & Analysis
- **Comprehensive metrics**: RMSE, MAE, R²
- **Model comparison** with automated best model selection
- **Feature importance analysis** for tree-based models
- **Performance visualization**
- **Automated submission file generation**

## 📁 Dataset

The dataset consists of 6 CSV files:

| File | Description | Key Columns |
|------|-------------|-------------|
| `train.csv` | Historical sales data | date, store_nbr, family, sales, onpromotion |
| `test.csv` | Test data for predictions | date, store_nbr, family, onpromotion |
| `stores.csv` | Store metadata | store_nbr, city, state, type, cluster |
| `holidays_events.csv` | Holiday information | date, type, locale, description |
| `oil.csv` | Daily oil prices | date, dcoilwtico |
| `transactions.csv` | Store transaction counts | date, store_nbr, transactions |

## 🚀 Installation

### Prerequisites
- Python 3.7+
- pip or conda package manager

### Required Packages
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Clone Repository
```bash
git clone https://github.com/yourusername/store-sales-forecasting.git
cd store-sales-forecasting
```

## 💻 Usage

### Basic Usage

```python
from store_sales_pipeline import preprocess_data, train_and_evaluate_pipeline

# Run complete preprocessing pipeline
train_final, test_final = preprocess_data()

# Train and evaluate all models
results, trained_models, data_splits = train_and_evaluate_pipeline(train_final)
```

### Advanced Usage

```python
# Custom preprocessing with specific transformations
train_processed = data_distribution(train_data, "log")
train_scaled = scale_data(train_processed, "standard")

# Individual model training
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)

# Feature importance analysis
feature_importance_analysis(trained_models, feature_names, top_n=20)

# Custom predictions
predictions = make_predictions(best_model, test_data, feature_columns)
```

### Configuration Options

```python
# Data transformation options
TRANSFORM_OPTIONS = ["log", "sqrt", "reciprocal", "yeo_johnson"]

# Scaling options  
SCALING_OPTIONS = ["minmax", "standard"]

# Model evaluation settings
CROSS_VALIDATION_FOLDS = 5
TEST_SIZE = 0.2
```

## 📈 Model Performance

### Evaluation Metrics

| Model | Test RMSE | Test R² | Test MAE | CV RMSE |
|-------|-----------|---------|----------|---------|
| GradientBoosting | 0.3842 | 0.9156 | 0.2103 | 0.3901 |
| RandomForest | 0.3956 | 0.9106 | 0.2187 | 0.4023 |
| ExtraTrees | 0.4012 | 0.9081 | 0.2234 | 0.4087 |
| VotingRegressor | 0.3798 | 0.9175 | 0.2076 | 0.3856 |
| Ridge | 0.4523 | 0.8734 | 0.2891 | 0.4612 |

*Note: Results may vary based on data preprocessing choices and random seeds*

### Key Insights
- **Best Performer**: VotingRegressor (ensemble of top 3 models)
- **Feature Importance**: Date features, store characteristics, and holiday indicators show highest importance
- **Cross-Validation**: Time series CV shows consistent performance across different time periods

## 🗂️ Project Structure

```
store-sales-forecasting/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── store_sales_pipeline.py           # Main pipeline script
│
├── data/                             # Data directory
│   ├── train.csv
│   ├── test.csv
│   ├── stores.csv
│   ├── holidays_events.csv
│   ├── oil.csv
│   └── transactions.csv
│
├── notebooks/                        # Jupyter notebooks
│   ├── data_exploration.ipynb
│   ├── feature_engineering.ipynb
│   └── model_analysis.ipynb
│
├── models/                           # Saved models
│   ├── best_model.pkl
│   └── model_metadata.json
│
├── results/                          # Output files
│   ├── submission.csv
│   ├── model_comparison.png
│   └── feature_importance.csv
│
└── utils/                           # Utility functions
    ├── visualization.py
    ├── evaluation.py
    └── preprocessing.py
```

## 🔧 Key Components

### 1. Data Preprocessing (`preprocess_data()`)
- Loads and merges all data sources
- Handles missing values and data types
- Creates comprehensive feature set
- Ensures train/test consistency

### 2. Feature Engineering (`modify_date()`)
- **Temporal features**: year, month, day, weekday
- **Cyclical encoding**: sine/cosine transformations for cyclical patterns
- **Holiday indicators**: national, regional, local holidays
- **External factors**: oil prices, transaction volumes

### 3. Model Training (`train_and_evaluate_pipeline()`)
- Trains multiple model types automatically
- Uses time-aware train/test splitting
- Performs time series cross-validation
- Selects best performing model

### 4. Evaluation System
- **Multiple metrics**: RMSE, MAE, R²
- **Time series validation**: Respects temporal order
- **Model comparison**: Automated ranking and selection
- **Visualization**: Performance plots and feature importance

## 📊 Results

### Performance Highlights
- **RMSE**: ~0.38 (best model)
- **R² Score**: ~0.92 (excellent fit)
- **Cross-validation**: Consistent performance across time periods
- **Feature Count**: 100+ engineered features

### Key Features Identified
1. **Date-based features**: Month, day of week, seasonality
2. **Store characteristics**: Type, cluster, location
3. **Holiday effects**: National and regional holidays
4. **Economic indicators**: Oil prices and trends
5. **Promotion effects**: Product promotion status

## 🛠️ Advanced Features

### Hyperparameter Tuning
```python
# Example grid search for RandomForest
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

# Use the hyperparameter_tuning_example() function
```

### Custom Transformations
```python
# Add custom feature transformations
def custom_feature_engineering(df):
    # Sales velocity
    df['sales_velocity'] = df.groupby('store_nbr')['sales'].pct_change()
    
    # Store-family interaction
    df['store_family_interaction'] = df['store_nbr'] * df['family_encoded']
    
    return df
```

## 🤝 Contributing

1. **Fork** the repository
2. **Create** your feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Contribution Guidelines
- Follow PEP 8 style guidelines
- Add unit tests for new functions
- Update documentation for new features
- Ensure all tests pass before submitting

## 📝 To-Do List

- [ ] Add deep learning models (LSTM, Transformer)
- [ ] Implement automated feature selection
- [ ] Add more sophisticated time series validation
- [ ] Create interactive dashboard for results
- [ ] Add model interpretability tools (SHAP, LIME)
- [ ] Implement automatic hyperparameter optimization

## ⚠️ Known Issues

- High-cardinality categorical variables may cause memory issues
- Some models may take significant time to train on large datasets
- Feature engineering pipeline may need adjustment for different datasets

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Kaggle](https://www.kaggle.com) for hosting the competition and providing the dataset
- [Scikit-learn](https://scikit-learn.org/) for the machine learning tools
- [Pandas](https://pandas.pydata.org/) for data manipulation capabilities
- The open-source community for various tools and libraries used

## 📞 Contact

- **Author**: Anurag Sarkar
- **Email**: sarkaranurag556@gmail.com


⭐ **If you found this project helpful, please give it a star!** ⭐
