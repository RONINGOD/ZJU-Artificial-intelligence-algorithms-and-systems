import pandas as pd
import sklearn
import numpy as np
from scipy.stats.mstats import winsorize
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
import joblib

version='v3'
train_data = pd.read_csv('./data/train_data.csv')
model_filename = f'./model_{version}.pkl'
imputer_filename = f'./imputer_{version}.pkl'
scaler_filename = f'./scaler_{version}.pkl'

def preprocess_data(data, imputer=None, scaler=None):
    
    column_name = ['Year', 'Life expectancy ', 'infant deaths', 'Alcohol',
               'percentage expenditure', 'Hepatitis B', 'Measles ', ' BMI ', 'under-five deaths ',
               'Polio', 'Total expenditure', 'Diphtheria ', ' HIV/AIDS', 'GDP', 'Population',
               ' thinness  1-19 years', ' thinness 5-9 years', 'Income composition of resources',
               'Schooling']
    data = data.drop(["Country", "Status"], axis=1)
    
    if imputer==None:
        imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
        imputer = imputer.fit(data[column_name])
    data[column_name] = imputer.transform(data[column_name])

    if scaler==None:
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
    data_norm = pd.DataFrame(scaler.transform(data), columns=data.columns)
    data_norm = data_norm.drop(
        ['infant deaths', 'Measles ', 'under-five deaths ', 'Population', 'Year'], axis=1)

    
    return data_norm, imputer, scaler

def model_fit(train_data):
 
    train_y = train_data.iloc[:, -1].values
    train_data = train_data.iloc[:, :-1]
 
    train_data_norm, imputer, scaler = preprocess_data(train_data)
 
    train_x = train_data_norm.values
 
    # 需要网格搜索的参数
    n_estimators = [i for i in range(650, 681, 5)]
    max_depth = [i for i in range(14, 18)]  # 最大深度
    min_samples_split = [i for i in range(2, 4)]  # 部节点再划分所需最小样本数
    min_samples_leaf = [i for i in range(3, 5)]  # 叶节点最小样本数
    max_samples = [i/100 for i in range(95, 97)]
    parameters = {'n_estimators': n_estimators,  # 弱学习器的最大迭代次数
                  'max_depth': max_depth,
                  'min_samples_split': min_samples_split,
                  'min_samples_leaf': min_samples_leaf,
                  'max_samples': max_samples
                  }
 
    regressor = RandomForestRegressor(
        bootstrap=True, oob_score=True, random_state=0)
    gs = RandomizedSearchCV(regressor, parameters, cv=10, refit=True, verbose=1, n_jobs=-1)
 
    gs.fit(train_x, train_y)
 
    joblib.dump(gs, model_filename)
    joblib.dump(imputer, imputer_filename)
    joblib.dump(scaler, scaler_filename)
 
    return gs

def predict(test_data):
    loaded_model = joblib.load(model_filename)
    imputer = joblib.load(imputer_filename)
    scaler = joblib.load(scaler_filename)
 
    test_data_norm, _, _ = preprocess_data(test_data, imputer, scaler)
    test_x = test_data_norm.values
    predictions = loaded_model.predict(test_x)
 
    return predictions

# data = pd.read_csv('./data/train_data.csv')
data = pd.read_csv('./kumarajarshi-life-expectancy-who/Life Expectancy Data.csv')
label = train_data.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, :-1], label, test_size = 0.20, random_state= 42)
train_data = pd.concat([x_train,y_train],axis=1)
test_data = pd.concat([x_test,y_test],axis=1)
model = model_fit(train_data)


print('最优参数: ', model.best_params_)
print('最佳性能: ', model.best_score_)
# label = train_data.loc[:,'Adult Mortality']
# data = train_data.iloc[:,:-1]
test_data_label = test_data.loc[:,'Adult Mortality']
test_data = test_data.iloc[:,:-1]
y_pred = predict(test_data)
# y_pred = predict(data)
r2 = r2_score(test_data_label, y_pred)
mse = mean_squared_error(test_data_label, y_pred)
print("MSE is {}".format(mse))
print("R2 score is {}".format(r2))

# Fitting 10 folds for each of 10 candidates, totalling 100 fits
# 最优参数:  {'n_estimators': 680, 'min_samples_split': 2, 'min_samples_leaf': 3, 'max_samples': 0.96, 'max_depth': 15}
# 最佳性能:  0.5488182157896306
# MSE is 4994.8452684867625
# R2 score is 0.6152498143847143