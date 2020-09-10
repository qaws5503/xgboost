import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

def cal_error(target,predict):
    error = []
    for num in range(len(y_test)):
        err = (target[num]-predict[num])/target[num]
        error.append(err)
    return error

def delete(data):
    new_data = data.copy()
    for row in range(1,len(data)):
        new_data[row] = data[row]-data[row-1]
    return new_data



data = pd.read_csv('/home/chiang/下載/GOOGL(1).csv')
# clear data 將爛資料去掉
data.dropna(how='any', inplace=True)
data = data.loc[:,['Year','Month','Date','Close']]
#data['Close'] = delete(data.loc[:,'Close'])

pre=[]
for day in range(2500,len(data)):
    X = data.copy()
    X.loc[:,'Close'] = X.loc[:,'Close'].shift(1)
    X['Close2'] = X.loc[:,'Close'].shift(2)
    X['Close3'] = X.loc[:,'Close'].shift(3)
    X['Close4'] = X.loc[:,'Close'].shift(4)
    X_train = X.iloc[:day,:].copy()
    y_train = data.iloc[:day,3].copy()
    
    X_train = X_train.drop(0)
    y_train = y_train.drop(0)
    
    X_test = X.iloc[day:day+1,:].copy()
    y_test = data.iloc[day:,3].copy()
# 將其於的資料再分成兩類：dtrain 跟 dtest
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
    xg_reg = xgb.XGBRegressor(objective='reg:linear', learning_rate=0.1, 
                              max_depth=50, n_estimators=500)
    
    xg_reg.fit(X_train, y_train)
    predictions = xg_reg.predict(X_test)
    pre.append(predictions)
y_test = data.iloc[2500:,3].copy()
plt.plot(np.array(pre))
plt.plot(y_test.to_numpy())

cal_error(y_test.to_numpy(),np.array(pre))
