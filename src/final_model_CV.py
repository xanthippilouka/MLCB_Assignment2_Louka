#Import required libraries

#Core scientific libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.base import clone
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold,  cross_val_score
from sklearn.metrics import (matthews_corrcoef, roc_auc_score, balanced_accuracy_score,f1_score, recall_score, precision_score, average_precision_score, confusion_matrix)
import optuna
import joblib
import shap
from sklearn.compose import ColumnTransformer

#Class for final hyperparameters tuning, the class is build for more estimators (not just one)
class CrossValidation:

    def __init__(self, data, estimators, parameter_space, n_folds=5,  seed=42): #cross-validation with 5 folds

        self.df = data
        self.target='num'
        self.estimators = estimators
        self.param_space = parameter_space
        self.n_folds = n_folds
        self.seed = seed

        self.quantitative = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        self.qualitative = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal", "ca"]

        self.X = data.drop(columns=[self.target])
        self.y= data[self.target]

        self.results = {} #Store the results from bootstraping
        self.best_params = {} #Store the winning hyperparameters from tuning

    #Function for handling the missing values, standardise scales (only on training data)
    #It is fixed only for the selected features for my model
    def preprocessing_pipeline(self, model, X):
        #Identify which features from the master lists are actually in X
        current_qual = [col for col in self.qualitative if col in X.columns]
        current_quant = [col for col in self.quantitative if col in X.columns]

        #Qualitative Pathway
        cat_pipe = Pipeline(steps=[
            ('impute', SimpleImputer(strategy="most_frequent")),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        #Quantitative Pathway
        num_pipe = Pipeline(steps=[
            ('impute', SimpleImputer(strategy="mean")),
            ('scale', RobustScaler())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', cat_pipe, current_qual),
                ('num', num_pipe, current_quant)
            ],
            remainder='drop'
        )

        #Return the full pipeline
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('clf', model)
        ])
   

    #Apply again Optuna for hyperparameters tuning to be consistent with the previous steps
    def optuna(self, X, y, estimator, param_space):
        def objective(trial):
            params = param_space(trial)
            #Cross validation - uses 5-fold CV to evaluate 50 different sets (from the 50 optuna trials) 
            cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

            trial_model = clone(estimator) #clone the estimator so it is untrained in every trial
            trial_model.set_params(**params) 
            
            pipeline = self.preprocessing_pipeline(trial_model, X)
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1_weighted')
            return scores.mean()
        
        study = optuna.create_study(direction="maximize", sampler =optuna.samplers.TPESampler(seed=self.seed))
        study.optimize(objective, n_trials=50) 
        return study.best_params
    
    def runfinalcv(self):
        #If I want to run more estimators
        for name, model in self.estimators:
            print(f"Optimizing {name}")

            #Determine Hyperparameters
            best_params = self.optuna(self.X, self.y, model, self.param_space[name]) 
            self.best_params[name] = best_params

            #CV - cross validation using the best hyperparameters
            cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
            fold_metrics = []

            for train_idx, val_idx in cv.split(self.X, self.y):
                X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
                y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]

                final_model = clone(model)
                final_model.set_params(**best_params)
                
                pipeline = self.preprocessing_pipeline(final_model, X_train)
                pipeline.fit(X_train, y_train)
                
                y_pred = pipeline.predict(X_val)

                if hasattr(pipeline, "predict_proba"):
                    probabilities = pipeline.predict_proba(X_val)
                    y_proba = probabilities[:,1]
                else:
                    y_proba = None
                
                fold_metrics.append(self.calculate_metrics(y_val, y_pred, y_proba))

            self.results[name] = pd.DataFrame(fold_metrics)
            print(f"Optimal {name} Parameters: {best_params}")
            
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
    
    def save_model(self, model_name, filename):
        #Save the model with the best hyperparameters
        #Save thee full pipeline not just the model (mathemtics)
        for name, model in self.estimators:
            if name == model_name:
                best_params = self.best_params[name]
                final_model = clone(model)
                final_model.set_params(**best_params)

                full_pipe = self.preprocessing_pipeline(final_model, self.X)
                full_pipe.fit(self.X, self.y) #Fit on the entire  set, only the winner algorithm
                #Save the full pipeline
                joblib.dump(full_pipe, filename)
                print(f"Complete pipeline for {model_name} saved successfully")

    def generate_SHAP(self, model_path, filename):
        
        #Load the trained model
        pipeline = joblib.load(model_path)

        #Seperate the steps of the pipeline
        preprocessor = pipeline.named_steps['preprocessor']
        model = pipeline.named_steps['clf']
    
        #Proprocess the raw data
        X_scaled = preprocessor.transform(self.X)
        feature_names = preprocessor.get_feature_names_out()
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

        #Initialize the Explainer with Background, Kernel SHAP needs a baseline to compare against,
        predict_pos_class = lambda x: model.predict_proba(x)[:, 1] 
        background = shap.kmeans(X_scaled_df, 5) #Use a clustering algorithm to summarize the dataset into 5 representative synthetics patients- average probability of heart disease across the entire dataset
        explainer = shap.KernelExplainer(predict_pos_class, background) #model-agnostic method
        
        #Calculate SHAP values
        actual_shap_values = explainer.shap_values(X_scaled_df)

        base_values_array = np.tile(explainer.expected_value, len(X_scaled_df))
        
        #Save SHAP values to CSV
        shap_df = pd.DataFrame(actual_shap_values, columns=feature_names)
        shap_df.to_csv(filename, index=False)
        print(f"SHAP values saved")

        #Create Explanation object 
        explanation = shap.Explanation(
            values=actual_shap_values, 
            base_values=base_values_array, 
            data=X_scaled_df, 
            feature_names=feature_names
        )

        #Summary Bar Plot -Shows the average absolute impact of each feature
        plt.figure()
        shap.plots.bar(explanation, max_display=25, show=False)
        plt.title(f"Global Feature Importance")
        plt.savefig("../figures/Task5/global_importance_bar.png", bbox_inches='tight')

        #Beeswarm Plot -Shows the impact (X-axis) and the feature value (color: red=high, blue=low)
        plt.figure()
        shap.plots.beeswarm(explanation, max_display=25, show=False)
        plt.title(f"Global Impact Distribution (Beeswarm)")
        plt.savefig("../figures/Task5/global_impact_beeswarm.png", bbox_inches='tight')

        plt.show()

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
def plot_metrics(stats_df, metrics = ['MCC', 'AUC', 'F1', 'BA', 'Precision', 'Recall'], filename="comparison_metrics.png"):

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

