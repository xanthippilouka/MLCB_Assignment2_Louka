#Import required libraries

#Core scientific libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, ttest_ind, mannwhitneyu, chi2_contingency
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')


features = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

map_disease = {0:'No-Disease', 1:'Disease'}
map_sex = {0:'Female', 1: 'Male'}
map_cp = {0:'typical angina', 1:'atypical angina', 2: 'non-anginal pain', 3: 'asymptomatic'}
map_fbs = {0:"False", 1: "True"}
map_restecg = {0:'Normal', 1:'ST-T wave abnormality', 2: 'left ventricular hypertrophy'}
map_slope = {0:'Upsloping', 1:'Flat', 2:';Downsloping'}
map_thal = {3:'Normal', 6:'Fixed defect', 7: 'Reversible defect'}


#Class for Data exploration
class DataExplorer:
    def __init__(self, data):
        self.df = data

        #Target variable 
        self.target = 'num'
        #Define quantitative and qualitative features 
        self.quantitative = ["age",  "trestbps", "chol",  "thalach", "oldpeak"]
        self.qualitative = [ "sex", "cp", "fbs", "restecg", "exang", "slope", "thal", "ca"]

    def basic_stats(self):
        print(f"Dataset Shape: {self.df.shape}")
        print(f"\n Dataset Information:")
        print(self.df.dtypes)
        print(f"\n Missing Values:")
        print(self.df.isnull().sum())
        print(f"\n Check for duplicates:")
        print(self.df.duplicated().sum())
        print(f"\n Descriptive Statistics")
        display(self.df.describe())

    def distributions_plots(self, plot_name=None):
        plt.figure(figsize=(15,10))

        for i, col in enumerate(self.quantitative, 1):
            plt.subplot(2,3,i)
            sns.histplot(self.df[col], kde=True, color='skyblue')
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
        plt.tight_layout()

        if plot_name:
            plt.savefig(f'../figures/Task1/{plot_name}', dpi=300, bbox_inches='tight')
        plt.show()

    def categorical_plots(self, plot_name=None):
        plt.figure(figsize=(15,12))

        for i, col in enumerate(self.qualitative + [self.target], 1):
            plt.subplot(4,3,i)
            sns.countplot(data=self.df, x=col, palette='viridis')
            plt.title(f'Count of {col}')
        plt.tight_layout()

        if plot_name:
            plt.savefig(f'../figures/Task1/{plot_name}', dpi=300, bbox_inches='tight')
        
        plt.show()

    def plot_boxplots(self, plot_name=None):
        plt.figure(figsize=(15,10))

        for i, col in enumerate(self.quantitative, 1):
            plt.subplot(2,3,i)
            sns.boxplot(data=self.df, x=self.target, y=col, palette='Set2')
            plt.title(f'{col} vs Heart Disease')
        plt.tight_layout()

        if plot_name:
            plt.savefig(f'../figures/Task1/{plot_name}', dpi=300, bbox_inches='tight')
        
        plt.show()

    def categorical_vs_target(self, plot_name=None):
        n_features = len(self.qualitative)
        rows = (n_features +1) // 2

        plt.figure(figsize=(15, rows * 5))

        for i, col in enumerate(self.qualitative, 1):
            plt.subplot(rows, 2, i)
            sns.countplot(data=self.df, x=col, hue=self.target, palette='magma')
            plt.title(f'{col} Distribution by Target')
            plt.legend(title='Heart Disease', labels=['No', 'Yes'])
        plt.tight_layout()

        if plot_name:
            plt.savefig(f'../figures/Task1/{plot_name}', dpi=300, bbox_inches='tight')
        
        plt.show()

class FeatureAnalysis:
    def __init__(self, df):
        #Target variable 
        self.target = 'num'
        #Define quantitative and qualitative features 
        self.quantitative = ["age",  "trestbps", "chol",  "thalach", "oldpeak"]
        self.qualitative = [ "sex", "cp", "fbs", "restecg", "exang", "slope", "thal", "ca"]

        #Handle the NA values by using the most frequent value in the feature
        imputer = SimpleImputer(strategy='most_frequent')
        self.new_df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        self.new_df = self.new_df.apply(pd.to_numeric)

    #Run some statistical tests first
    def run_statistical_tests(self):

        for col in self.quantitative:
            #Group data by target
            f0 = self.new_df[self.new_df[self.target]==0][col].dropna()
            f1 = self.new_df[self.new_df[self.target]==1][col].dropna()

            #Check for normality
            _, p_normf0 = shapiro(f0)
            _, p_normf1 = shapiro(f1)

            if p_normf0 > 0.05 and p_normf1 > 0.05:
                #Parametric test
                test = "T-Test"
                _, p_val = ttest_ind(f0,f1)
            else:
                #Non-Parametric
                test = "Mann-Whitney U"
                _, p_val = mannwhitneyu(f0,f1)

            significance = "*" if p_val < 0.05 else ""
            print(f"{col:<12} | {test:<15} | {p_val:<10.4f} | {significance}")

        for col in self.qualitative:
            if col == self.target: continue
            tab = pd.crosstab(self.new_df[col], self.new_df[self.target])
            _, p_val, _, _ = chi2_contingency(tab)
            significance = "*" if p_val < 0.05 else ""
            print(f"{col:<12} | {'Chi-Square':<15} | {p_val:<10.4f} | {significance}")

    def correlation_analysis(self, plot_name=None):

        df_encoded = self.new_df.copy()
        qual_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal", "ca"]
        
        for col in qual_cols:
            if col in df_encoded.columns:
                df_encoded[col] = df_encoded[col].astype('category').cat.codes
        
        #Use Spearman Correlation (Ideal for mixed data types) (non-parametric)
        corr = df_encoded.corr(method='spearman')
        
        mask = np.triu(np.ones_like(corr, dtype=bool))
        #plot with heatmap
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', center=0, annot_kws={"size":7})
        plt.title("Correlation Analysis of Features")

        if plot_name:
            plt.savefig(f'../figures/Task1/{plot_name}', dpi=300, bbox_inches='tight')
        
        plt.show()

def apply_pca(self, plot_name=None):
    
    qual_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal", "ca"]
    quant_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    
    X = self.new_df.drop(columns=[self.target])
    
    #Preprocessing for PCA
    #StandardScaler and OneHotEncoder
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False), qual_cols),
            ('num', StandardScaler(), quant_cols)
        ]
    )
    
    # Transform the data
    X_transformed = preprocessor.fit_transform(X)

    #Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_transformed)
    var = pca.explained_variance_ratio_

    #Visualization
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=X_pca[:, 0], 
        y=X_pca[:, 1], 
        hue=self.new_df[self.target], 
        palette='Set1', 
        s=60,
        alpha=0.8
    )
    
    plt.title(f"PCA: Full Feature Set (Variance Explained: {sum(var):.2%})")
    plt.xlabel(f"PC1 ({var[0]:.2%})")
    plt.ylabel(f"PC2 ({var[1]:.2%})")
    
    if plot_name:
        plt.savefig(f'../figures/Task1/{plot_name}', dpi=300, bbox_inches='tight')
    plt.show()
    


    






