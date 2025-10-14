import pandas as pd
from typing import Dict, Any, Tuple
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args


def optimize_xgboost_cv(
    X: pd.DataFrame,
    y: pd.Series,
    param_bounds: Dict[str, Tuple[float, float]] = None,
    cv_folds: int = 5,
    n_iterations: int = 50,
    n_initial_points: int = 10,
    scoring: str = 'neg_mean_absolute_error',
    random_state: int = 42,
    verbose: bool = True,
    early_stopping_rounds: int = None,
    early_stopping_threshold: float = 1e-4
) -> Tuple[Dict[str, Any], float, pd.DataFrame]:
    """
    Optimize XGBoost hyperparameters using Bayesian optimization with cross-validation.
    Uses scikit-optimize library for efficient Bayesian optimization.

    Args:
        X: Feature matrix
        y: Target variable
        param_bounds: Parameter bounds. Uses defaults if None.
        cv_folds: Number of cross-validation folds
        n_iterations: Number of optimization iterations
        n_initial_points: Number of random initialization points
        scoring: Scoring metric for cross-validation
        random_state: Random seed
        verbose: Whether to print progress
        early_stopping_rounds: Stop if no improvement after this many iterations
        early_stopping_threshold: Minimum improvement to reset early stopping counter

    Returns:
        Tuple of (best_params, best_score, optimization_history)
    """
    import xgboost as xgb
    from sklearn.model_selection import cross_val_score

    if param_bounds is None:
        param_bounds = {
            'max_depth': (3, 10),
            'learning_rate': (0.01, 0.3),
            'n_estimators': (50, 500),
            'min_child_weight': (1, 10),
            'subsample': (0.5, 1.0),
            'colsample_bytree': (0.5, 1.0),
            'gamma': (0.0, 5.0),
            'reg_alpha': (0.0, 1.0),
            'reg_lambda': (0.0, 1.0)
        }

    space = [
        Integer(int(param_bounds['max_depth'][0]), int(param_bounds['max_depth'][1]), name='max_depth'),
        Real(param_bounds['learning_rate'][0], param_bounds['learning_rate'][1], name='learning_rate', prior='log-uniform'),
        Integer(int(param_bounds['n_estimators'][0]), int(param_bounds['n_estimators'][1]), name='n_estimators'),
        Integer(int(param_bounds['min_child_weight'][0]), int(param_bounds['min_child_weight'][1]), name='min_child_weight'),
        Real(param_bounds['subsample'][0], param_bounds['subsample'][1], name='subsample'),
        Real(param_bounds['colsample_bytree'][0], param_bounds['colsample_bytree'][1], name='colsample_bytree'),
        Real(param_bounds['gamma'][0], param_bounds['gamma'][1], name='gamma'),
        Real(param_bounds['reg_alpha'][0], param_bounds['reg_alpha'][1], name='reg_alpha'),
        Real(param_bounds['reg_lambda'][0], param_bounds['reg_lambda'][1], name='reg_lambda')
    ]

    iteration_counter = [0]

    @use_named_args(space)
    def objective(**params):
        """Objective function for Bayesian optimization."""
        iteration_counter[0] += 1

        params_model = {
            'max_depth': int(params['max_depth']),
            'learning_rate': params['learning_rate'],
            'n_estimators': int(params['n_estimators']),
            'min_child_weight': int(params['min_child_weight']),
            'subsample': params['subsample'],
            'colsample_bytree': params['colsample_bytree'],
            'gamma': params['gamma'],
            'reg_alpha': params['reg_alpha'],
            'reg_lambda': params['reg_lambda'],
            'objective': 'reg:squarederror',
            'random_state': random_state,
            'enable_categorical': True
        }

        model = xgb.XGBRegressor(**params_model)
        scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring, n_jobs=-1)
        score = scores.mean()

        if verbose:
            print(f"Iteration {iteration_counter[0]}/{n_iterations}: score={score:.4f}")
            if iteration_counter[0] > n_initial_points:
                print(f"  Params: max_depth={params['max_depth']}, lr={params['learning_rate']:.4f}, n_est={params['n_estimators']}")

        return -score

    callbacks = []
    if early_stopping_rounds is not None:
        from skopt.callbacks import DeltaYStopper
        stopper = DeltaYStopper(delta=early_stopping_threshold, n_best=early_stopping_rounds)
        callbacks.append(stopper)

    result = gp_minimize(
        objective,
        space,
        n_calls=n_iterations,
        n_initial_points=n_initial_points,
        random_state=random_state,
        verbose=False,
        callback=callbacks if callbacks else None
    )

    best_params = {
        'max_depth': int(result.x[0]),
        'learning_rate': float(result.x[1]),
        'n_estimators': int(result.x[2]),
        'min_child_weight': int(result.x[3]),
        'subsample': float(result.x[4]),
        'colsample_bytree': float(result.x[5]),
        'gamma': float(result.x[6]),
        'reg_alpha': float(result.x[7]),
        'reg_lambda': float(result.x[8])
    }

    best_score = -result.fun

    history_data = []
    for x_vals, y_val in zip(result.x_iters, result.func_vals):
        history_data.append({
            'max_depth': int(x_vals[0]),
            'learning_rate': float(x_vals[1]),
            'n_estimators': int(x_vals[2]),
            'min_child_weight': int(x_vals[3]),
            'subsample': float(x_vals[4]),
            'colsample_bytree': float(x_vals[5]),
            'gamma': float(x_vals[6]),
            'reg_alpha': float(x_vals[7]),
            'reg_lambda': float(x_vals[8]),
            'score': -y_val
        })

    history = pd.DataFrame(history_data)

    if verbose and early_stopping_rounds is not None and len(result.x_iters) < n_iterations:
        print(f"\nEarly stopping triggered after {len(result.x_iters)} iterations")

    return best_params, best_score, history
