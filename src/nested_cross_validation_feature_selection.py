#Import required libraries

#Core scientific libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import sklearn

from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_val_score, StratifiedKFold
from sklearn.metrics import (matthews_corrcoef, roc_auc_score, balanced_accuracy_score,f1_score, recall_score, precision_score, average_precision_score, confusion_matrix)
import optuna
import joblib
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from functools import partial


class NestedCrossValidation:
    #I will use this class for both baseline comparison (with default hyperparameters) and for hyperparameters tuning (Optuna)
    #I will set this class to work on two modes - optimize=False no inner loop (training on the outer loop) and optimize=True (inner loop for Optuna for hyperparameters tuning)
    #Add another mode option regarding the feature selection (feature_selection = False - no feature selection, feature_selection = True - feature selection)
    def __init__(self, data, estimators, parameter_space, R=10, n_outer=5, n_inner=3, seed=42, optimize=False, feature_selection = False):

        self.df = data
        self.target='num'
        self.quantitative = ["age",  "trestbps", "chol",  "thalach", "oldpeak"]
        self.qualitative = [ "sex", "cp", "fbs", "restecg", "exang", "slope", "thal", "ca"]

        self.estimators = estimators
        self.param_space = parameter_space
        
        self.n_rounds = R #Rounds of cross validation, the class will run 10 rounds --> Distribution of scores (CI)
        self.n_outer = n_outer
        self.n_inner = n_inner
        self.seed = seed
        self.optimize = optimize
        self.feature_selection = feature_selection

        self.X = data.drop(columns=[self.target])
        self.y = data[self.target]

        self.results = {}
        self.feature_stability = {}
        

    #Function for handling the missing values, standardise scales
    def preprocessing_pipeline(self, model, X):
        current_qual = [col for col in self.qualitative if col in X.columns]
        current_quant = [col for col in self.quantitative if col in X.columns]

        #pipeline for Qualitative  features
        cat_pipe = Pipeline(steps=[
            ('impute', SimpleImputer(strategy="most_frequent")),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        #pipeline for Quantitative features
        num_pipe = Pipeline(steps=[
            ('impute', SimpleImputer(strategy="mean")), # 'mean' or 'median' is standard for numbers
            ('scale', RobustScaler())
        ])

        #ColumnTransformer 
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', cat_pipe, current_qual),
                ('num', num_pipe, current_quant)
            ],
            remainder='drop' 
        )

        #Final Pipeline: Preprocessor - Classifier
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('clf', model)
        ])

    def runcv(self):
        
        #Use RepeatedStratifiedKFold to ensure proper class imbalance handling
        outer_cv = RepeatedStratifiedKFold(n_splits=self.n_outer, n_repeats=self.n_rounds, random_state=self.seed) #Outer folds

        #Mask for discrete features 
        is_discrete = [col in self.qualitative for col in self.X.columns]

        total_runs = self.n_rounds * self.n_outer

        for name, model in self.estimators:
            print(f"Processing: {name}")
            fold_metrics = []

            #Store the features
            self.feature_stability[name] = {col: 0 for col in self.X.columns}

            for train_index, test_index in outer_cv.split(self.X, self.y):
                fold_model = clone(model)
                X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
                y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
                
                if self.feature_selection:
                    #Apply imputer because The MI feature selection method cannot handle NA
                    temp_imputer = SimpleImputer(strategy="most_frequent")
                    X_train_imputed = pd.DataFrame(temp_imputer.fit_transform(X_train), columns=X_train.columns)

                    #Perfom feature selection using Mutual Information (MI)
                    mutual_function = partial(mutual_info_classif, discrete_features=is_discrete, random_state = self.seed)
                    selector = SelectKBest(score_func=mutual_function, k=7)
                    selector.fit(X_train_imputed, y_train)

                    #identify chosen features
                    cols = selector.get_support(indices=True)
                    selected_names = self.X.columns[cols]

                    for feature in selected_names:
                        self.feature_stability[name][feature] += 1

                    #Reduce the initial split sets for only the selected features
                    X_train = X_train[selected_names]
                    X_test = X_test[selected_names]
                else:
                    X_train = X_train
                    X_test = X_test
                
                should_optimize = (self.optimize==True) and (self.param_space is not None)

                if should_optimize and (name in self.param_space):
                    #Enter the Inner loop for Optuna tuning
                    best_params = self.optuna(X_train, y_train, fold_model, self.param_space[name]) #clone : create a new, unfitted version of the model 
                    fold_model.set_params(**best_params)
                
                #If not enter the inner loop for optimization, training and evaluation on the outer loop
                #First apply preprocessing pipeline on training set only
                pipeline = self.preprocessing_pipeline(fold_model, X_train)

                pipeline.fit(X_train, y_train)

                #Make predictions on test set
                y_pred = pipeline.predict(X_test)

                if hasattr(pipeline, "predict_proba"):
                    probabilities = pipeline.predict_proba(X_test)
                    y_proba = probabilities[:,1]
                else:
                    y_proba = None

                fold_metrics.append(self.calculate_metrics(y_test, y_pred, y_proba))

            self.results[name] = pd.DataFrame(fold_metrics) #At the end of one round I have 5 sets of the metrics for each model. After the 10 rounds, I have 50 sets of metrics
 
            if self.feature_selection:
                #Generate a DataFrame with the selected features metrics
                stability_df = pd.DataFrame(list(self.feature_stability[name].items()), columns=['Feature', 'Count'])
                
                #Add the frequency (divide with 50)
                stability_df['Frequency'] = stability_df['Count'] / 50
                stability_df = stability_df.sort_values(by='Frequency', ascending=False)
                
                # Save to CSV
                filename = f"../data/Task4/stability_{name}.csv"
                stability_df.to_csv(filename, index=False)
                
                print(f"Stability Table saved")

    def optuna(self, X_train_outer, y_train_outer, estimator, param_space):
        def objective(trial):
            params = param_space(trial)
            #Cross validation tecnhiques for the inner loop
            inner_cv = StratifiedKFold(n_splits=self.n_inner, shuffle=True, random_state=self.seed) #Inner folds
            scores = []

            for train_idx, val_idx in inner_cv.split(X_train_outer, y_train_outer):
                X_train_inner, X_val_inner = X_train_outer.iloc[train_idx], X_train_outer.iloc[val_idx]
                y_train_inner, y_val_inner = y_train_outer.iloc[train_idx], y_train_outer.iloc[val_idx]

                trial_model = clone(estimator) 
                trial_model.set_params(**params) #set the parameters to the model, unpack the dictionary
                 
                pipeline_inner = self.preprocessing_pipeline(trial_model, X_train_inner)
                pipeline_inner.fit(X_train_inner, y_train_inner)
                y_pred_inner = pipeline_inner.predict(X_val_inner)

                #Optimization target
                scores.append(f1_score(y_val_inner, y_pred_inner, average = 'weighted'))

            return np.mean(scores)

        study = optuna.create_study(direction="maximize", sampler =optuna.samplers.TPESampler(seed=self.seed)) #make the sampler behave in a deterministic way
        study.optimize(objective, n_trials=50)
        return study.best_params

    def calculate_metrics(self, y_test, y_pred, y_proba):

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn/(tn+fp) if (tn +fp) >0 else 0

        return {
            'MCC': matthews_corrcoef(y_test, y_pred),
            'AUC': roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan,
            'BA' : balanced_accuracy_score(y_test, y_pred),
            'F1' :f1_score(y_test, y_pred, average='weighted'),
            'Recall': recall_score(y_test, y_pred),
            'Precision' : precision_score(y_test, y_pred),
            'PRAUC' : average_precision_score(y_test, y_proba) if y_proba is not None else np.nan,
            'Specificity': specificity
            }
    

       
#Function for Bootstrap
def bootstrap(results_df, n_resamples=1000, confidence_level=0.95, seed=42):
    np.random.seed(seed) #starting point
    resamples = n_resamples

    #Initialize the empty statistics matrix
    summary_stats = []

    for metric in results_df.columns:
        scores = results_df[metric].values
        n=len(scores)
        bootstrap_medians =[]
        
        for i in range(resamples):
            #Pick random indices with replacement
            idx = np.random.choice(np.arange(n), size=n, replace=True)
            resample_scores = scores[idx]

            bootstrap_medians.append(np.median(resample_scores))

        #Calculate Confidence Intervals
        lower_bound = (1-confidence_level) /2
        upper_bound = 1 - lower_bound

        ci_lower = np.percentile(bootstrap_medians, lower_bound * 100)
        ci_upper = np.percentile(bootstrap_medians, upper_bound *100)
        median = np.median(scores)

        summary_stats.append({
            'Metric' : metric,
            'Median' : median,
            'CI_Lower' : ci_lower,
            'CI_Upper': ci_upper
        })

    return pd.DataFrame(summary_stats)


#Generate a function to save the results from the models after bootstraping
def get_models_statistics(results_dict, output_filename="stats_report.csv"):
    all_sum = []

    for model_name, df in results_dict.items():
        summary = bootstrap(df)
        summary['Model'] = model_name
        all_sum.append(summary)

    #Save the report
    report_df = pd.concat(all_sum).reset_index(drop=True)
    report_df.to_csv(output_filename, index=False)

    return report_df


#Function to plot metrics
def plot_metrics(stats_df, metrics = ['MCC', 'Recall', 'AUC'], filename="comparison_metrics.png"):

    plt.figure(figsize=(15,5))

    for i, metric in enumerate(metrics):
        plt.subplot(1, len(metrics), i+1)
        data = stats_df[stats_df['Metric']==metric]

        #Median and 95% intervals
        plt.errorbar(
            x=data['Model'],
            y=data['Median'],
            yerr=[data['Median'] - data['CI_Lower'], data['CI_Upper'] - data['Median']],
            fmt = 'o', color='teal', capsize=5, capthick=2
        )
        plt.title(f"Median {metric} (95% CI)", fontsize=12)
        plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
        plt.ylabel('Score')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout(pad=3.0)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

