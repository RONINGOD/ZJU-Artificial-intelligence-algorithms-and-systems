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
import xgboost as xgb

version='v4_xgboost'
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
        [ 'Year'], axis=1)

    
    return data_norm, imputer, scaler

def model_fit(train_data):
 
    train_y = train_data.iloc[:, -1].values
    train_data = train_data.iloc[:, :-1]
 
    train_data_norm, imputer, scaler = preprocess_data(train_data)
 
    train_x = train_data_norm.values
    # XGBoost模型
    # model = xgb.XGBRegressor(n_estimators=100,max_depth=10)
    model = xgb.XGBRFRegressor(n_estimators=1000,max_depth=20)
    # # 设置参数搜索空间
    # n_estimators = [i for i in range(100,200,5)]
    # max_depth = [i for i in range(3,30)]
    # param_grid = {'n_estimators': n_estimators,
    #               'max_depth': max_depth}
    
    # gs = GridSearchCV(model, param_grid, cv=10, n_jobs=-1, verbose=1)

    # gs.fit(train_x, train_y)
    model.fit(train_x,train_y)
    
    # 保存最佳模型
    # joblib.dump(gs.best_estimator_, model_filename) 
    joblib.dump(model,model_filename)
    joblib.dump(scaler, scaler_filename)
    joblib.dump(imputer, imputer_filename)
 
    # return gs
    return model

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

label = data.loc[:, 'Adult Mortality']
data_drop_na = ~np.isnan(label)
data = data[data_drop_na]
label = data.loc[:, 'Adult Mortality']
data_no_label = data.drop(['Adult Mortality'],axis=1)
x_train, x_test, y_train, y_test = train_test_split(data_no_label, label, test_size = 0.20, random_state= 42)
train_data = pd.concat([x_train,y_train],axis=1)
test_data = pd.concat([x_test,y_test],axis=1)
print(train_data.shape)
model = model_fit(train_data)
# print('最优参数: ', model.best_params_)
# print('最佳性能: ', model.best_score_)
# label = data.loc[:,'Adult Mortality']
# data = data.iloc[:,:-1]
origin_data = pd.read_csv('./data/train_data.csv')
test_data_label = origin_data.loc[:,'Adult Mortality']
test_data = origin_data.drop(['Adult Mortality'],axis=1)
# test_data_label = test_data.loc[:,'Adult Mortality']
# test_data = test_data.iloc[:,:-1]

y_pred = predict(test_data)
r2 = r2_score(test_data_label, y_pred)
mse = mean_squared_error(test_data_label, y_pred)
print("MSE is {}".format(mse))
print("R2 score is {}".format(r2))

# 最优参数:  {'max_depth': 3, 'n_estimators': 105}
# 最佳性能:  0.5117272751951878
# MSE is 2848.3688301236334
# R2 score is 0.816786323997798
