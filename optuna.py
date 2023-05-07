import optuna
import catboost as cb


# General syntax
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=500)

# Plotting
fig = optuna.visualization.plot_param_importances(study)
fig.show()

# Objective with config
study_objective = lambda trial: objective(trial, model_config)

# Warm start: adding trials
study.enqueue_trial({"hidden_size": 64, 
                    "dropout": 0.05,
                    "weight_decay": 1e-05})


# Multi-target
study = optuna.create_study(directions=["maximize", "maximize"])
fig = optuna.visualization.plot_pareto_front(study, target_names = ["mse", "mae"])

fig = optuna.visualization.plot_param_importances(study, target = lambda t: t.values[1], target_name="mae")
fig = optuna.visualization.plot_parallel_coordinate(study, target = lambda t: t.values[1], target_name="mae")


# Catboost
def objective(trial):
    model = cb.CatBoostRegressor(
        iterations=2000,
        l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1e-07, 100, log=True),
        subsample=trial.suggest_float('subsample', 0.2, 1),
        random_strength=trial.suggest_float('random_strength', 0.1, 5),
        depth=trial.suggest_int('depth', 1, 8),
        min_data_in_leaf=trial.suggest_int('leaf', 1, 200),
        rsm=trial.suggest_float('rsm', 0.2, 1),
        random_seed=42,
        loss_function="RMSE",
        eval_metric="AUC",
        thread_count=50)
    
    model.fit(
        train_features[feature_names], 
        train_target,
        eval_set=(valid_features[feature_names], valid_target),
        verbose=0
    )
    
    prediction = model.predict(valid_features[feature_names])
    metric = roc_auc_score(valid_target, prediction)
    
    return metric
