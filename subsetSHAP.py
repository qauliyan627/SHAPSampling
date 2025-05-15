import math
import random
import time

from scipy.optimize import minimize
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
import xgboost as xgb

class importData: # 資料導入
    def __init__(self):
        # import dataset
        uciDataset = fetch_ucirepo(id=186) 
        # Feature Engineering
        X = uciDataset.data.features
        y = uciDataset.data.targets
        # Number of features(M)
        self.features = X.columns
        self.featureNum = len(self.features)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.20, random_state=0)
        # predict data
        self.X_predictData = self.X_test.iloc[[0]] # 要解釋的資料
        self.midData = pd.DataFrame([self.X_test.mean()])
    
class Model(): # 製作模型和輸出預測值
    def __init__(self):
        impoData = importData()
        self.X_train = impoData.X_train
        self.y_train = impoData.y_train
        self.createModel()
        
    def createModel(self):
        # Modelling
        # Train model
        self.model = xgb.XGBRegressor(objective="reg:squarederror")
        self.model.fit(self.X_train, self.y_train)
        print('model ok')

    # predict
    def predict(self, data):
        predictResult = self.model.predict(data)[0]
        return predictResult

# transiton: h()
def transFun(z):
    impoData = importData()
    transData = impoData.X_predictData
    features = impoData.features
    midData = impoData.midData
    for i in range(len(z)):
        if z[i] == 0:
            transData.iloc[0][features[i]] = midData.iloc[0][features[i]]
    return transData

# Kernel Weight: Pi_x()
def weightFunc(subsetSize):
    weightNum = (featureNum-1) / (math.comb(featureNum,subsetSize)*subsetSize*(featureNum-subsetSize))
    #print(weightNum)
    return weightNum

def objective_function(variables):
    total_sum = 0
    for i in sampList:
        i_bin = format(i, 'b')
        i_bin = i_bin.zfill(featureNum)
        transData = transFun(i_bin)
        tempTerm = midPredict # phi_0
        for k in range(featureNum):
            if i_bin[k] == '1':
                tempTerm += variables[k] # phi_1~phi_n
        term = ((model.predict(transData) - (tempTerm))**2)*weightFunc(i_bin.count('1'))
        total_sum += term
    return total_sum

def equality_constraint(variables):
    equaTerm = midPredict
    for i in range(featureNum):
        equaTerm += variables[i]
    equaTerm -= ansPredict
    return equaTerm  # 所有shapley value相加等於預測結果

def ScipyMinimize():
    initial_guess = np.arange(featureNum)
    constraints = ({'type': 'eq', 'fun': equality_constraint})
    options = {'maxiter': 10000}
    result = minimize(objective_function, initial_guess, constraints=constraints, method='SLSQP') # 計算SHAP值
    return result

def sampling():
    impoData = importData()
    featureNum = impoData.featureNum
    while True:
        temp = random.randint(1, 2**featureNum-1)
        if temp not in sampList:
            sampList.append(temp)
            break
    return sampList

def getGap(optimal_variables):
    gap = 0
    for i in range(featureNum):
        gap += abs(ANS_LIST[i] - optimal_variables[i])
    return gap

ANS_LIST = [-0.07932989864247109, -0.07938996574848867, -0.0793634085821533, -0.0793892396058209, -0.07938264416386287, -0.07928658076617318, -0.0792978789151697, -0.07935561937152125, -0.07938196742866044, -0.07937382088674427, -0.07936567434473574]
ROUND_LIMIT = 50 # 最大循環次數
SAMPLING_LIMIT = 32 # 最大抽樣數量
impoData = importData()
model = Model()
featureNum = impoData.featureNum
midPredict = model.predict(impoData.midData)
ansPredict = model.predict(impoData.X_predictData)
# 第一回
sampList = []
sampList.append(random.randint(1, 2**featureNum-1))
print(f"sampList: {sampList}")
result = ScipyMinimize()
if result.success:
    minimum_value = result.fun
    optimal_variables = result.x
    maxX = max(optimal_variables)
    minX = min(optimal_variables)
    width = maxX - minX
    
    print(f"round 1")
    featureStr = f"對應的變數值: x0 = {optimal_variables[0]}"
    for i in range(1, featureNum): 
        featureStr += f", x{i} = {optimal_variables[i]}"
    print(featureStr)
    print(f"SHAPwidth = {width}")
    print(f"gap = {getGap(optimal_variables)}")
# 第二回之後
for i in range(2, ROUND_LIMIT):
    print(f"round {i}")
    sampList = sampling()
    print(f"sampList: {sampList}")
    
    time_start = time.time()
    result = ScipyMinimize()
    time_end = time.time()
    print(f"計算時間:{time_end-time_start}")
    
    if result.success:
        minimum_value = result.fun
        optimal_variables = result.x
        maxX = max(optimal_variables)
        minX = min(optimal_variables)
        newWidth = maxX - minX
        if newWidth > width:
            del sampList[-1]
            continue
        else:
            width = newWidth
        
        featureStr = f"對應的變數值: x0 = {optimal_variables[0]}"
        for i in range(1, featureNum): 
            featureStr += f", x{i} = {optimal_variables[i]}"
        print(featureStr)
        print(f"SHAPwidth = {width}")
        print(f"gap = {getGap(optimal_variables)}")
    if len(sampList) >= SAMPLING_LIMIT:
        break