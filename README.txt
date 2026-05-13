Goal of the Assignment: Development of a machine learning algorithm to predict and classify correctly the patients with and without heart disease.

Python version 3.13.2 was used in the present study.

Execution steps:
    1. Add the `students_dataset.csv` file in the `data/` folder
    2. Open and run all the cells of the following notebook files, in order
        `notebooks/data_exploration.ipynb`
        `notebooks/nestedCV_tuning.ipynb`
        `notebooks/nestedCV_feature_selection.ipynb`
        `notebooks/final_model.ipynb`

Structure:
    >data/ : contains the initial dataset .csv files (students_dataset.csv)
        -Task3/ : contains generated report for baseline models' metrics, .csv files with the 50 values for each metric for each model after tuning, .csv file for the median plus 95% CI od these models' metrics
        -Task4/ : .csv file with features stability metrics, .csv file with the median and 95% CI from GNB model after feature selection, .csv file with GNB model's metrics before and after feature selection
        -Task5/ : .csv file with the GNB model's final metrics, .csv file with the SHAP values
    >figures/ : contains all the figures generated for the different Tasks
        -Task1/ : Figures for Task 1 - Exploration Data Analysis
        -Task2/ : Figures for Task 2 - Nested Cross-Validation Diagram
        -Task3/ : Figures for Task 3 - Studied models' hyperparameters tuning
        -Task4/ : Figures for Task 4 - GNB Feature Selection
        -Task5/ : Figures for Task 5 - GNB final training and SHAP values
    >models/ : saved winner algorithm (GNB_final.pkl)
    >notebooks/ : Include all the notebooks in which the pipeline for the assignment Tasks is executed
        -data_exploration : Explore the dataset (dimensions, type, distribution/statistics of the variables, correlation matrix, PCA), Generation of Respective Plots
        -nestedCV_tuning : Perform repeated nestedCV for the baseline models (default hyperparameters) and for baseline models with Optuna hyperparameters tuning
        -nestedCV_feature selection : Perform repeated nestedCV after feature selection for the winner algorithm (GNB)
        -final_model : Train the winner algorithm in the initial dataset and generate SHAP values 
    >src/ :
        -functions.py : DataExplorer class (EDA), FeatureAnalysis class (NA handling, statistical tests, correlation analysis, PCA)
        -nested_cross_validation.py : Repeated Nested Cross-Validation Class (outer 5-fold loop + inner 3-fold loop)
        -nested_cross_validation_feature_selection.py : Updated nested_cross_validation.py to include the feature selection pipeline
        -final_model_CV.py : 5-fold Cross-validation for hyperparameter tuning using optuna(with 5-fold CV), function for saving the model pipeline and generate SHAP values 
    requirements.txt : include the libraries used in the present study (intall in every python notebook)
*data folder not commited to GitHub (.gitignore)