import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error,r2_score


def scoring_regression(regressor,X_train,y_train,y_test,y_pred,cross_val=True):
    """
    Scoring regression solutions

    Using target values y_train and the predicted target values, the score for each scoring is returned as a dataframe
    """

    scoring = {
        'r2': 'r2',
        'mse': make_scorer(mean_squared_error),
        'rmse': make_scorer(mean_squared_error, squared=False),
        'mae': make_scorer(mean_absolute_error),
        'mape': make_scorer(lambda y_tar, y_pred: np.mean(np.abs((y_tar - y_pred) / y_tar)) * 100)
    }
    
    # Perform cross-validation and obtain average scores
    cv_average_scores = {}
    for key, value in scoring.items():
        cv_scores = cross_val_score(regressor, X=X_train, y=y_train, scoring=value, cv=StratifiedKFold(shuffle=True, random_state=23333))
        cv_average_scores[key] = cv_scores.mean()

    # Create a DataFrame from the average_scores dictionary
    cv_scores_df = pd.DataFrame({'Evaluation Metric': cv_average_scores.keys(), 'Cross Validation Score': cv_average_scores.values()})

    # Calculate R2 score, mean squared error, root mean squared error, mean absolute error, mean absolute percentage error

    r2_cv = r2_score(y_test, y_pred)
    mse_cv = mean_squared_error(y_test, y_pred)
    rmse_cv = mean_squared_error(y_test, y_pred, squared=False)
    mae_cv = mean_absolute_error(y_test, y_pred)
    mape_cv = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    model_scores_df = pd.DataFrame({
        'Model Score': [r2_cv, mse_cv, rmse_cv, mae_cv, mape_cv]
    })

    # Concatenate the DataFrames vertically
    scores_df = pd.concat([cv_scores_df, model_scores_df], axis=1)

    # Reset the index
    scores_df.reset_index(drop=True, inplace=True)

    return scores_df
    # Display the combined DataFrame
    # print(combined_df)
