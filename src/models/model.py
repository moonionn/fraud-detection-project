"""
Model Training Functions
"""

import lightgbm as lgb
import xgboost as xgb
from src.models.sampling import get_sampler


def train_model(train_data, model_type: str, use_sampling: bool, sampling_type: str, **model_params):
    """
    Train model with specified type and sampling method

    Args:
        train_data: Training data
        model_type: Model type (lgb, xgb, lr, rf, svm, knn)
        use_sampling: Whether to use sampling
        sampling_type: Sampling type (none, undersample, oversample, smote, smoteenn)
        **model_params: Additional model parameters

    Returns:
        model: Trained model
        feature_cols: Feature columns used
    """
    exclude_cols = ['TransactionID', 'isFraud', 'TransactionDT']
    feature_cols = [col for col in train_data.columns if col not in exclude_cols]

    X = train_data[feature_cols].copy()
    y = train_data['isFraud'].copy()

    if use_sampling:
        sampler = get_sampler(sampling_type)
        X, y = sampler(X, y)
        print(f"  Applied sampling: {sampling_type}")

    if model_type == 'lgb':
        model = model_lightgbm(X, y, **model_params)
    elif model_type == 'xgb':
        model = model_xgboost(X, y, **model_params)
    elif model_type == 'rf':
        model = model_random_forest(X, y, **model_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    print(f'  Model trained with {len(feature_cols)} features')
    return model, feature_cols


def model_lightgbm(X_train, y_train, **params):
    """Train LightGBM model"""
    default_params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'random_state': 42
    }
    default_params.update(params)

    train_set = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(
        default_params,
        train_set,
        num_boost_round=100,
        callbacks=[lgb.log_evaluation(period=0)]
    )
    return model


def model_xgboost(X_train, y_train, **params):
    """Train XGBoost model"""
    default_params = {
        'n_estimators': 100,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': 42,
        'enable_categorical': True
    }
    default_params.update(params)

    model = xgb.XGBClassifier(**default_params)
    model.fit(X_train, y_train, verbose=False)
    return model

def model_random_forest(X_train, y_train, **params):
    """Train Random Forest model"""
    from sklearn.ensemble import RandomForestClassifier

    default_params = {
        'n_estimators': 100,
        'random_state': 42,
        'n_jobs': -1
    }
    default_params.update(params)

    model = RandomForestClassifier(**default_params)
    model.fit(X_train, y_train)
    return model


def predict_model(model, test_data, feature_cols, model_type='lgb'):
    """
    Make predictions using trained model

    Args:
        model: Trained model
        test_data: Test data
        feature_cols: Feature columns
        model_type: Model type (lgb, xgb, lr, rf, svm, knn)

    Returns:
        y_pred_proba: Predicted probabilities
        y_pred: Predicted labels
    """
    X_test = test_data[feature_cols]

    if model_type == 'lgb':
        y_pred_proba = model.predict(X_test)
    else:
        y_pred_proba = model.predict_proba(X_test)[:, 1]

    y_pred = (y_pred_proba >= 0.5).astype(int)
    return y_pred_proba, y_pred