from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

def use_xgboost(X_train,X_test,y_train,y_test):
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
        print ("SCORE:", accuracy)
        return {'loss': -accuracy, 'status': STATUS_OK }

    trials = Trials()

    best_hyperparams = fmin(fn = objective,
                            space = space,
                            algo = tpe.suggest,
                            max_evals = 100,
                            trials = trials)
    
    print("The best hyperparameters are : ","\n")
    print(best_hyperparams)