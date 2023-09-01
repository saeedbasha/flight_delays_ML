import pandas as pd
from xgboost import XGBClassifier,XGBRegressor
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from scripts import scoring

from sklearn.metrics import accuracy_score, recall_score, precision_score,mean_squared_error
from sklearn.ensemble import AdaBoostRegressor


def use_xgboost_classification(X_train,X_test,y_train,y_test):
    """
    Using XG-Boost for predictions
    """

    space={'max_depth': hp.quniform("max_depth", 2,18, 1),
        'gamma': hp.uniform ('gamma', 0,9),
        'reg_alpha' : hp.uniform('reg_alpha', 0,10),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': 99,
        'seed': 0,
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.3)
    }

    def objective(space):
        clf=XGBClassifier(
                        objective='binary:logistic',
                        n_estimators =space['n_estimators'], 
                        max_depth = int(space['max_depth']), 
                        gamma = space['gamma'],
                        reg_alpha = int(space['reg_alpha']),
                        min_child_weight=int(space['min_child_weight']),
                        colsample_bytree=int(space['colsample_bytree']),
                        eval_metric="auc",  # Set eval_metric here
                        )
        
        evaluation = [( X_train, y_train), ( X_test, y_test)]
        
        clf.fit(X_train, y_train,
            eval_set=evaluation,  # Provide evaluation dataset
            early_stopping_rounds=10)
        
        pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, pred>0.5)
        # print ("SCORE:", accuracy)
        return {'loss': -accuracy, 'status': STATUS_OK }

    trials = Trials()

    best_hyperparams = fmin(fn = objective,
                            space = space,
                            algo = tpe.suggest,
                            max_evals = 100,
                            trials = trials)
    
    # Extract the best hyperparameters
    # best_n_estimators = best_hyperparams['n_estimators'] # I didn't vary this, so the output doesn't have it
    best_max_depth = best_hyperparams['max_depth']
    best_gamma = best_hyperparams['gamma']
    best_colsample_bytree = best_hyperparams['colsample_bytree']
    best_min_child_weight = best_hyperparams['min_child_weight']
    best_reg_alpha = best_hyperparams['reg_alpha']
    best_reg_lambda = best_hyperparams['reg_lambda']

    # Create the best model using the best hyperparameters
    best_model = XGBClassifier(
        n_estimators=99,
        max_depth=int(best_max_depth),
        gamma=best_gamma,
        reg_alpha= best_reg_alpha,
        reg_lambda= best_reg_lambda,
        min_child_weight = best_min_child_weight,
        colsample_bytree = best_colsample_bytree,
        eval_metric="auc"  # Set eval_metric here
    )

    # Fit the best model on your training data
    best_model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    # Create a DataFrame
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Recall', 'Precision'],
        'Score': [accuracy, recall, precision]
    })

    return [best_hyperparams,metrics_df]

def use_xgboost_regression(X_train,X_test,y_train,y_test):
    """
    Using XG-Boost for predictions
    """

    space={
        'max_depth': hp.quniform("max_depth", 2,18, 1),
        'gamma': hp.uniform ('gamma', 0,9),
        'reg_alpha' : hp.uniform('reg_alpha', 0,10),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': 100,
        'seed': 0,
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.3)
    }

    def objective(space):
        clf=XGBRegressor(
                        n_estimators =space['n_estimators'], 
                        max_depth = int(space['max_depth']), 
                        gamma = space['gamma'],
                        reg_alpha = int(space['reg_alpha']),
                        min_child_weight=int(space['min_child_weight']),
                        colsample_bytree=int(space['colsample_bytree']),
                        eval_metric="auc",  # Set eval_metric here
                        )
        
        evaluation = [( X_train, y_train), ( X_test, y_test)]
        
        clf.fit(X_train, y_train,
            eval_set=evaluation,  # Provide evaluation dataset
            early_stopping_rounds=10)
        
        y_pred = clf.predict(X_test)
        # Use mean squared error as the loss function
        mse = mean_squared_error(y_test, y_pred)
        # print ("SCORE:", accuracy)
        return {'loss': mse, 'status': STATUS_OK }

    trials = Trials()

    best_hyperparams = fmin(fn = objective,
                            space = space,
                            algo = tpe.suggest,
                            max_evals = 100,
                            trials = trials)
    
    # Extract the best hyperparameters
    # best_n_estimators = best_hyperparams['n_estimators'] # I didn't vary this, so the output doesn't have it
    best_max_depth = best_hyperparams['max_depth']
    best_gamma = best_hyperparams['gamma']
    best_colsample_bytree = best_hyperparams['colsample_bytree']
    best_min_child_weight = best_hyperparams['min_child_weight']
    best_reg_alpha = best_hyperparams['reg_alpha']
    best_reg_lambda = best_hyperparams['reg_lambda']

    # Create the best model using the best hyperparameters
    best_model = XGBRegressor(
        n_estimators=99,
        max_depth=int(best_max_depth),
        gamma=best_gamma,
        reg_alpha= best_reg_alpha,
        reg_lambda= best_reg_lambda,
        min_child_weight = best_min_child_weight,
        colsample_bytree = best_colsample_bytree,
        eval_metric="auc"  # Set eval_metric here
    )

    # Fit the best model on your training data
    best_model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = best_model.predict(X_test)

    results = scoring.scoring_regression(best_model,X_train,y_train,y_test,y_pred,cross_val=True)
    print(results)
    return results

def use_AdaBoost(X_train,X_test,y_train,y_test):
    regr = AdaBoostRegressor(random_state=0, n_estimators=100)
    regr.fit(X_train, y_train)
    # regr.predict(X_test)
    return [regr.score(X_train,y_train),regr.score(X_test,y_test)]