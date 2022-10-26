import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor, XGBRFRegressor
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold


class DataSetRegression():
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X.reset_index(drop=True)
        self.y = y
        self.X_transform = pd.DataFrame([])
        self._standart_scaler = StandardScaler()
        self.data = pd.concat([self.X, self.y], axis=1)
        self._ordinalEncoder = OrdinalEncoder()
        
    def show_miss_data(self):
        miss_data = self.X.isnull().sum().sort_values()
        miss_data /= self.X.shape[0]
        if len(miss_data[miss_data > 0]) == 0:
            print('There is no missing data')
            return None
        miss_data[miss_data > 0].plot.bar(color='green')
        return miss_data[miss_data > 0].index
        
    def fill_na_mode(self, columns):
        for i in columns:
            self.X[i] = self.X[i].fillna(self.X[i].mode()[0])
            
    def one_hot_encoder(self, columns, drop_first=True):
        self.X[columns] = self.X[columns].astype('str')
        self.X_transform = pd.concat([pd.get_dummies(self.X[columns], drop_first=drop_first), self.X_transform], axis=1)
        
    def add_without_scale(self, columns):
        self.X_transform = pd.concat([self.X[columns], self.X_transform], axis=1)
    
    def standart_scale(self, columns):
        sk = pd.DataFrame(self._standart_scaler.fit_transform(self.X[columns]), columns=columns)
        self.X_transform = pd.concat([sk, self.X_transform], axis=1)
    
    def train_test_split(self, *args, **kwargs):
        return train_test_split(self.X_transform[:len(self.y)], self.y, *args, **kwargs)
    @property
    def submission_data(self):
        return self.X_transform[len(self.y):]
    @property
    def all_train_data(self):
        return self.X_transform[:len(self.y)]
    
    def ordinal_encoder(self, columns):
        self.X[columns] = self.X[columns].astype('str')
        gg = self._ordinalEncoder.fit_transform(self.X[columns])
        oe = pd.DataFrame(gg, columns=columns)
        self.X_transform = pd.concat([self.X_transform, oe], axis=1)
    
    def correlation_matrix(self, n2show=25):
        corrM = self.data.corr()
        fig, ax = plt.subplots(dpi = 150, figsize=(10, 8))
        cols = corrM.nlargest(n2show, self.y.name)[self.y.name].index
        sns.heatmap(corrM.loc[cols, cols], annot=True, cmap="YlGnBu", linewidths=0.1,)
        return cols
    
    def quant(self, col='label', q=0.99):
        if col == 'label':
            quant = np.quantile(self.y, 0.99)
            self.y[self.y > quant] = quant
        else:
            quant = np.quantile(self.X[col], q)
            self.X.loc[self.X[col] > quant, col] = quant
            
    def show_quantile(self, col):
        d = pd.DataFrame.quantile(self.X[col], [0.0, 0.25, 0.5, 0.75, 1])
        d = d - d.iloc[2]
        d = d.append(d.iloc[0]/d.iloc[1], ignore_index=True)
        d = d.append(d.iloc[4]/d.iloc[3], ignore_index=True)
        d = d.set_index(pd.Index(['0%', '25%', '50%', '75%', '100%', '0/25', '1/75']))
        d = d.replace([np.inf, -np.inf], np.nan)
        return d.dropna(axis=1)