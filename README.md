# Used Car Price Prediction - Data Science Nigeria Hackathon

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Feature Engineering](#feature-engineering)
  - [Basic Features](#basic-features)
  - [Interaction Features](#interaction-features)
  - [Luxury Segmentation](#luxury-segmentation)
  - [Frequency Encoding](#frequency-encoding)
  - [Category Encoding](#category-encoding)
- [Model Architecture](#model-architecture)
  - [LightGBM Models](#lightgbm-models)
  - [CatBoost Model](#catboost-model)
- [Ensemble Strategy](#ensemble-strategy)
- [Performance Results](#performance-results)
- [Technical Implementation](#technical-implementation)
- [Usage Instructions](#usage-instructions)
- [Dependencies](#dependencies)

## Project Overview

This repository contains the solution for the Data Science Nigeria 2025 Used Car Price Prediction hackathon. The approach combines advanced feature engineering with ensemble modeling using LightGBM and CatBoost algorithms to predict used car prices with high accuracy.

## Dataset Description

The dataset contains the following features for used car price prediction:

| Feature | Type | Description |
|---------|------|-------------|
| `id` | Object | Unique identifier |
| `brand` | Object | Car manufacturer |
| `model` | Object | Car model |
| `model_year` | Integer | Manufacturing year |
| `milage` | Integer | Car mileage |
| `fuel_type` | Object | Type of fuel |
| `engine` | Object | Engine specifications |
| `transmission` | Object | Transmission type |
| `ext_col` | Object | Exterior color |
| `int_col` | Object | Interior color |
| `accident` | Object | Accident history |
| `clean_title` | Object | Title status |
| `price` | Float | Target variable (car price) |

## Feature Engineering

### Basic Features

**Car Age Calculation**
```python
train['car_age'] = 2025 - train['model_year']
test['car_age'] = 2025 - test['model_year']
```

### Interaction Features

**Engine-Transmission Interaction**
- `engine_transmission`: Combination of engine and transmission type
- `color_combo`: Exterior and interior color combination
- `model_ext_color`: Model and exterior color interaction
- `brand_model`: Brand and model combination

### Luxury Segmentation

**Luxury Brand Classification**
- Luxury brands: `['Porsche', 'Mercedes-Benz', 'BMW', 'Bentley', 'Lamborghini', 'Land']`
- Binary flags: `is_luxury_brand`, `is_regular_brand`

**Luxury Model Classification**
- High-end models identified through outlier analysis
- Includes models like '911 Carrera S', 'AMG G 63 Base', 'Corvette Stingray'
- Binary flags: `is_luxury_model`, `is_regular_model`

**Advanced Feature Interactions**
- High mileage flag (>200,000 miles)
- Model decade grouping
- Luxury-mileage interactions
- Brand-decade combinations

### Frequency Encoding

Categorical variables are enhanced with frequency encoding to capture the rarity/popularity of each category:

```python
for col in obj_cols:
    combined = pd.concat([train[col], test[col]], axis=0)
    freq_map = combined.value_counts(normalize=False).to_dict()
    train[col + "_freq"] = train[col].map(freq_map)
    test[col + "_freq"] = test[col].map(freq_map)
```

### Category Encoding

Consistent categorical encoding across train and test sets using pandas categorical codes.

## Model Architecture

### LightGBM Models

**Model Configuration**
```python
lgb_params = {
    'subsample': 0.8,
    'reg_lambda': 0.1,
    'reg_alpha': 0.1,
    'num_leaves': 50,
    'n_estimators': 800,
    'max_depth': 5,
    'learning_rate': 0.01,
    'colsample_bytree': 0.7,
    'objective': 'regression',
    'metric': 'rmse'
}
```

**Feature Sets**
- **LGB Model 1**: Focus on brand-model interactions and luxury flags
- **LGB Model 2**: Emphasis on individual brand/model features

### CatBoost Model

**Configuration**
```python
cat_params = {
    'loss_function': 'RMSE',
    'learning_rate': 0.05,
    'depth': 6,
    'iterations': 500,
    'subsample': 0.8,
    'colsample_bylevel': 0.8
}
```

## Ensemble Strategy

**Inverse RMSE Weighting**

The ensemble uses inverse RMSE weighting to combine predictions from multiple models:

```python
def weighted_ensemble(predictions, rmses):
    weights = 1 / np.array(rmses)
    weights /= weights.sum()
    ensemble_pred = np.sum([w * p for w, p in zip(weights, predictions)], axis=0)
    return ensemble_pred, weights
```

Three ensemble combinations are evaluated:
1. LightGBM Model 1 + CatBoost
2. LightGBM Model 2 + CatBoost  
3. All three models combined

## Performance Results

| Model | RMSE | Performance |
|-------|------|-------------|
| LightGBM Model 1 | 67,776.37 | Individual |
| LightGBM Model 2 | 67,775.95 | Individual |
| CatBoost | 67,814.97 | Individual |
| **Ensemble 1** (LGB1+Cat) | **67,716.84** | Best 2-model |
| **Ensemble 2** (LGB2+Cat) | **67,718.91** | Alternative 2-model |
| **Ensemble 3** (All models) | **67,711.61** | **Best Overall** |

**Key Insights:**
- Ensemble methods consistently outperform individual models
- The 3-model ensemble achieves the lowest RMSE of 67,711.61
- Balanced weights across models (~33% each) indicate complementary strengths

## Technical Implementation

**Data Preprocessing Pipeline**
1. Feature engineering and interaction creation
2. Luxury segmentation based on domain knowledge
3. Frequency encoding for categorical variables
4. Consistent encoding across train/test splits

**Model Training Strategy**
1. 80/20 train-validation split
2. Individual model training with optimized hyperparameters
3. Validation-based ensemble weight calculation
4. Final prediction generation

**Evaluation Methodology**
- Primary metric: Root Mean Square Error (RMSE)
- Validation strategy: Hold-out validation
- Ensemble evaluation: Inverse RMSE weighting

## Usage Instructions

1. **Data Preparation**
   ```python
   # Load and prepare your data
   train = pd.read_csv('train.csv')
   test = pd.read_csv('test.csv')
   ```

2. **Feature Engineering**
   ```python
   # Apply all feature engineering steps
   train = add_features(train)
   test = add_features(test)
   ```

3. **Model Training**
   ```python
   # Train individual models
   lgb_model_1.fit(X_train[lgb_features_1], y_train)
   lgb_model_2.fit(X_train[lgb_features_2], y_train)
   cat_model.fit(X_train[cat_features], y_train)
   ```

4. **Generate Predictions**
   ```python
   # Create ensemble predictions
   final_predictions = weighted_ensemble(predictions, rmses)
   ```

## Dependencies

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor
```

**Required Packages:**
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- lightgbm >= 3.3.0
- catboost >= 1.0.0

---

**Author**: Adediran Adeyemi
**Competition**: Data Science Nigeria Hackathon 2025 (Used Car Price Prediction)  
**Final Score**: RMSE = 67,711.61
