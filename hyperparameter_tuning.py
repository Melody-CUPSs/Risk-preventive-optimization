# ============================================================
# Hyperparameter tuning for RF, LightGBM, and XGBoost metamodels
# Search ranges follow Table S20 in Supplemental Materials
# ============================================================

# ---------------------- RF ----------------------
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# 1. Define the hyperparameter search space (Table S20)
param_grid_rf = {
    "n_estimators": [50, 100, 200, 300, 500],
    "max_depth": [3, 5, 8, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5, 10],
    "max_features": ["sqrt", "log2", 0.7, 0.8],
    "max_samples": [0.7, 0.8, 0.9, None],
    "criterion": ["squared_error", "absolute_error"]
}

# 2. Initialize the Random Forest regressor
rf = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)

# 3. Perform grid search with five-fold cross-validation
grid_rf = GridSearchCV(
    estimator=rf,
    param_grid=param_grid_rf,
    scoring="r2",
    cv=5,
    n_jobs=-1,
    verbose=1
)

# 4. Train the model and retrieve the best hyperparameters
grid_rf.fit(X_train, y_train)
best_model_rf = grid_rf.best_estimator_
best_params_rf = grid_rf.best_params_
print("RF Best hyperparameters:", best_params_rf)
print("RF Best cross-validation R2:", grid_rf.best_score_)


# ---------------------- LightGBM ----------------------
from lightgbm import LGBMRegressor

# 1. Define the hyperparameter search space (Table S20)
param_grid_lgbm = {
    "n_estimators": [100, 200, 300, 500],
    "num_leaves": [7, 15, 31, 63],
    "max_depth": [3, 5, 7, -1],
    "learning_rate": [0.01, 0.05, 0.1],
    "feature_fraction": [0.6, 0.7, 0.8, 0.9],
    "bagging_fraction": [0.6, 0.7, 0.8, 0.9],
    "bagging_freq": [1, 5],
    "min_child_samples": [5, 10, 20, 30],
    "reg_alpha": [0, 0.1, 1, 10],
    "reg_lambda": [0, 0.1, 1, 10]
}

# 2. Initialize the LightGBM regressor
lgbm = LGBMRegressor(
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

# 3. Perform grid search with five-fold cross-validation
grid_lgbm = GridSearchCV(
    estimator=lgbm,
    param_grid=param_grid_lgbm,
    scoring="r2",
    cv=5,
    n_jobs=-1,
    verbose=1
)

# 4. Train the model and retrieve the best hyperparameters
grid_lgbm.fit(X_train, y_train)
best_model_lgbm = grid_lgbm.best_estimator_
best_params_lgbm = grid_lgbm.best_params_
print("LightGBM Best hyperparameters:", best_params_lgbm)
print("LightGBM Best cross-validation R2:", grid_lgbm.best_score_)


# ---------------------- XGBoost ----------------------
from xgboost import XGBRegressor

# 1. XGBoost hyperparameter search space (Table S20)
param_grid_xgb = {
    "n_estimators": [100, 300, 500, 800],
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "reg_alpha": [0, 0.1, 1, 10],
    "reg_lambda": [0, 0.1, 1, 10],
    "min_child_weight": [1, 3, 5, 8],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
    "subsample": [0.6, 0.7, 0.8, 0.9]
}

# 2. Initialize the XGBoost regressor
xgb = XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

# 3. Grid search with five-fold cross-validation
grid_xgb = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid_xgb,
    scoring="r2",
    cv=5,
    n_jobs=-1,
    verbose=1
)

# 4. Train and retrieve the best hyperparameters
grid_xgb.fit(X_train, y_train)
best_model_xgb = grid_xgb.best_estimator_
best_params_xgb = grid_xgb.best_params_
print("XGBoost Best hyperparameters:", best_params_xgb)
print("XGBoost Best cross-validation R2:", grid_xgb.best_score_)
