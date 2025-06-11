import os
import math
import random
import time
import logging

from scipy.stats import qmc
from scipy.optimize import minimize
import numpy as np
import pandas as pd

import shap
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
import xgboost as xgb

class Model():
    def __init__(self):
        # Modelling
        # Train model
        self.model = xgb.XGBRegressor(objective="reg:squarederror")
        self.model.fit(X_train, y_train)
        print('model ok')
        
    def getAnsAndMidPredict(self):
        ansPredict = self.predict(X_predictData)
        midPredict = self.predict(midData)
        return ansPredict, midPredict

    # predict
    def predict(self, data):
        prediction_pandas = self.model.predict(data)[0]
        #print(f"predict: {prediction_pandas}")
        return prediction_pandas

def _setData(): # import dataset
    uciDataset = fetch_ucirepo(id=ID[DATASET])
    # Feature Engineering
    X = uciDataset.data.features
    y = uciDataset.data.targets
    cat_columns = X.select_dtypes(['object']).columns
    if len(cat_columns) > 0:
        for cc in cat_columns:
            codes, uniques = pd.factorize(X[cc])
            X[cc] = codes
    cat_columns = y.select_dtypes(['object']).columns
    if len(cat_columns) > 0:
        for cc in cat_columns:
            codes, uniques = pd.factorize(y[cc])
            y[cc] = codes
    print(X)
    print(y)
    
    ## predictors only
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.20,
                                                        random_state=0)
    return X_train, X_test, y_train, y_test

def _getExactShapValue(model):
    explainer = shap.TreeExplainer(model, X_train)
    shap_values = explainer(X_predictData)
    np.savetxt(f"{LOCATION}\\ANS\\ans_{EXPLAIN_DATA}.txt", shap_values[0].values)
    return shap_values[0].values

def _h(z): # transiton: h()
    transData = X_predictData.copy(deep=True)
    for i in range(len(z)):
        if z[i] == '0':
            transData.loc[:, columns[i]] = midData.loc[0, columns[i]]
    return transData

def _Pi_x(subsetSize): # Kernel Weight: Pi_x()
    weightNum = (featureNum-1) / (math.comb(featureNum,subsetSize)*subsetSize*(featureNum-subsetSize))
    #print(weightNum)
    return weightNum

def objective_function(variables):
    total_sum = 0
    for i in range(1,2**featureNum-1):
        if i not in samplingList:
            continue
        i_bin = format(i, 'b')
        i_bin = i_bin.zfill(featureNum)
        transData = _h(i_bin)
        tempTerm = midPredict
        for k in range(featureNum):
            if i_bin[k] == '1':
                tempTerm += variables[k]
        if i_bin in binToAnsDict.keys():
            predictAns = binToAnsDict[i_bin]
        else:
            predictAns = model.predict(transData)
            binToAnsDict[i_bin] = predictAns
        term = ((predictAns - (tempTerm))**2)*_Pi_x(i_bin.count('1'))
        total_sum += term
    return total_sum

def equality_constraint(variables):
    equaTerm = midPredict
    for i in range(featureNum):
        equaTerm += variables[i]
    equaTerm -= ansPredict
    return equaTerm  # 所有shapley value相加等於預測結果

def minimizeFunc():
    initial_guess = np.arange(featureNum)
    constraints = ({'type': 'eq', 'fun': equality_constraint})
    options = {'maxiter': 10000}
    time_start = time.time()
    result = minimize(objective_function, initial_guess, constraints=constraints, method='SLSQP') # 計算SHAP值
    time_end = time.time() # SHAP值計算結束時間
    print(f"計算時間={time_end - time_start}")
    return result

def getLoss(optimal_variables): # 取得跟精準SHAP值的差距
    loss = 0
    for i in range(featureNum):
        loss += abs(ANS_LIST[i] - optimal_variables[i])
    print("loss in getLoss:",loss)
    return loss

def randomSampling(samplingNum): #  mode0: 隨機
    tempList = []
    while True:
        r = random.randint(1, 2**featureNum-1)
        if r in tempList:
            continue
        else:
            tempList.append(r)
        if len(tempList) >= samplingNum:
            break
    return tempList

def randPairSampling(samplingNum): # mode1: 隨機配對抽樣
    tempList = []
    while True:
        r = random.randint(1, 2**featureNum//2-1)
        if r in tempList: continue
        else: tempList.append(r)
        if len(tempList) >= samplingNum//2: break
    # 反向配對
    for i in tempList:
        tempStr = ""
        i_bin = format(i, 'b')
        i_bin = i_bin.zfill(featureNum)
        # 反向二進位 01交換
        for j in range(featureNum):
            if i_bin[j] == '0': tempStr+='1'
            else : tempStr+='0'
        i_r = int(tempStr,2)
        if i_r not in tempList: tempList.append(i_r)
    return tempList

def sobolSampling(samplingNum): # mode2: 低差異序列: Sobol
    x = 2**featureNum - 2
    while True:
        sobol = qmc.Sobol(d=1)  # 1-dimensional
        samples = sobol.random(n=samplingNum).ravel()  # Generate samples
        int_samples = (samples * x + 1).astype(int)
        for i in range(len(int_samples)):
            for j in range(i+1, len(int_samples)):
                if int_samples[i] == int_samples[j]:
                    int_samples[i] += np.random.choice([-1, 1])
        if len(set(int_samples)) != samplingNum : continue
        else: break
    return int_samples

def haltonSampling(samplingNum): # mode3: 低差異序列: Halton
    x = 2**featureNum - 2
    while True:
        sobol = qmc.Halton(d=1)  # 1-dimensional
        samples = sobol.random(n=samplingNum).ravel()  # Generate samples
        int_samples = (samples * x + 1).astype(int)
        for i in range(len(int_samples)):
            for j in range(i+1, len(int_samples)):
                if int_samples[i] == int_samples[j]:
                    int_samples[i] += np.random.choice([-1, 1])
        if len(set(int_samples)) != samplingNum : continue
        else: break
    return int_samples

def pairedSampling(): # mode4: 凸型配對(左右對稱)
    midNum = 2**featureNum//2
    temp = 0
    tempList = [midNum]
    # 抽樣
    i = 2
    while True:
        temp = midNum + fibonacci(i)
        if temp >= 2**featureNum: break
        tempList.append(temp)
        i+=1
    # 反向配對
    for i in tempList:
        tempStr = ""
        i_bin = format(i, 'b')
        i_bin = i_bin.zfill(featureNum)
        # 反向二進位 01交換
        for j in range(featureNum):
            if i_bin[j] == '0': tempStr+='1'
            else : tempStr+='0'
        i_r = int(tempStr,2)
        if i_r not in tempList: tempList.append(i_r)
    return tempList

def ldFibSampling(samplingNum): # mode5: 費氏數列 + 低差異序列想法(挑選最大區間抽樣) + 配對抽樣
    top = 2**featureNum//2-1
    but = 2**featureNum
    tempList = [top, but]
    n_top = top
    n_but = but
    for _ in range(samplingNum//2):
        # 計算最大可用費氏數
        ran = n_but - n_top - 1
        maxFib = 0
        while True:
            if ran < fibonacci(maxFib):
                maxFib -= 1
                break
            else: maxFib += 1
        # 抽樣
        while True:
            ranFib = fibonacci(random.randint(1, maxFib))
            if n_top + ranFib not in tempList:
                tempList.append(n_top + ranFib)
                break
        tempList.sort()
        # 找尋最大間距
        maxRange = 0
        for i in range(1, len(tempList)):
            t_top = tempList[i-1]
            t_but = tempList[i]
            if maxRange < t_but-t_top:
                n_top = t_top
                n_but = t_but
                maxRange = t_but-t_top
    tempList.remove(top)
    tempList.remove(but)
    for i in tempList:
        tempStr = ""
        i_bin = format(i, 'b')
        i_bin = i_bin.zfill(featureNum)
        # 反向二進位 01交換
        for j in range(featureNum):
            if i_bin[j] == '0': tempStr+='1'
            else : tempStr+='0'
        i_r = int(tempStr,2)
        if i_r not in tempList: tempList.append(i_r)
    tempList.sort()
    return tempList

def fibonacci(n): # 計算費氏數
    if n == 0: return 0
    elif n == 1: return 1
    elif n < 0: return -1
    if n in fibonacciSeq.keys():
        return fibonacciSeq[n]
    else:
        fn = fibonacci(n-1)
        fm = fibonacci(n-2)
        fibonacciSeq[n] = fn + fm
        return fn + fm

def sampling(sampling_num, mode=0): # 選擇抽樣方法
    time_start = time.time() # 開始計算時間
    if sampling_num == "COMP_MODE":
        mode = COMP_MODE
    if mode == 0: 
        samplingList = randomSampling(sampling_num)
    elif mode == 1:
        samplingList = randPairSampling(sampling_num)
    elif mode == 2:
        samplingList = sobolSampling(sampling_num)
    elif mode == 3:
        samplingList = haltonSampling(sampling_num)
    elif mode == 4:
        samplingList = pairedSampling()
    elif mode == 5:
        samplingList = ldFibSampling(sampling_num)
    logging.info("結束抽樣")

    time_end = time.time() # 抽樣結束時間
    samplingTime = time_end - time_start # 計算抽樣時間
    samplingList.sort()
    print(f"samplingTime={samplingTime}")
    print(f"samplingList={samplingList}")
    print(f"len: {len(samplingList)}")
    return samplingList
    
def getSpac(spList): # 取得抽樣結果中各元素的距離
    spacList = []
    for i in range(1,len(spList)):
        spacList.append(abs(spList[i] - spList[i-1]))
    return spacList

def saveLossSampList(sampList):
    with open(f"{LOCATION}ALLsamplingLossList.txt", 'a') as f:
        np.savetxt(f,sampList)

def getLOSS(): # 計算LOSS_LIMIT(COMP_MODE)
    result = minimizeFunc()
    if result.success:
        optimal_variables = result.x
        loss = getLoss(optimal_variables) # 取得和精準SHAP值之間的差距
        LOSS_LIMIT[EXPLAIN_DATA] = loss
        np.save(f"{LOCATION}\\LOSS\\loss_mode{COMP_MODE}.npy", LOSS_LIMIT)
        np.savetxt(f"{LOCATION}\\LOSS\\LossShapValue_{EXPLAIN_DATA}.txt", optimal_variables)
        return np.load(f"{LOCATION}\\LOSS\\loss_mode{COMP_MODE}.npy", allow_pickle=True).item()
    else:
        print(f"優化失敗: {result.message}")

def mainFunc():
    global LOSS_LIMIT
    global samplingList
    global avgAll
    global countAll
    samplingList = []
    time_total = 0
    sampling_time_total = 0
    loss_total = 0
    loss_max = 0
    loss_min = 9999
    count = 0
    
    allLossList = []
    allSampList = []
    allSpacList = [] # 抽選子集組的各子集距離
    allShapValue = [] # 記錄每次計算的SHAP值
    
    for j in range(ROUND):
        print(f"LOOPNUM_{LOOPNUM}, ROUND_{j}, MODE{MODE},ID{ID[DATASET]}")
        
        # samplingList: 特徵子集抽樣 array = 1~2**featureNum-1
        print(f"SAMPLING_NUM = {SAMPLING_NUM[DATASET]}")
        samplingList = sampling(SAMPLING_NUM[DATASET], MODE)
        
        initial_guess = np.arange(featureNum)
        constraints = ({'type': 'eq', 'fun': equality_constraint})
        options = {'maxiter': 10000}
        time_start = time.time()
        result = minimize(objective_function, initial_guess, constraints=constraints, method='SLSQP') # 計算SHAP值
        time_end = time.time() # SHAP值計算結束時間
        if result.success:
            minimum_value = result.fun
            optimal_variables = result.x
            
            print(f"找到最小值: {minimum_value}")
            featureStr = f"對應的變數值: x0 = {optimal_variables[0]}"
            resultTemp = []
            resultTemp.append(optimal_variables[0])
            for i in range(1, featureNum): 
                resultTemp.append(optimal_variables[i])
                featureStr += f", x{i} = {optimal_variables[i]}"
            print(featureStr)
            print(f"中間預測值: {midPredict}")
            
            allSampList.append(samplingList)# 保存全部子集組合的抽樣結果
            allSpacList.append(getSpac(samplingList))# 保存全部子集組合的子集間距離
            allShapValue.append(resultTemp) # 保存所以的ShapValue
            
            loss = getLoss(optimal_variables) # 取得和精準SHAP值之間的差距
            if LOSS_LIMIT.get(EXPLAIN_DATA, -1) == -1:
                LOSS_LIMIT = getLOSS()
            if loss < LOSS_LIMIT[EXPLAIN_DATA]: count+=1 # 計算差距小於設定值的次數
            if loss > loss_max: loss_max = loss
            if loss < loss_min: loss_min = loss
            allLossList.append(loss)
            loss_total += loss
            print(f"差距: {loss}")
            time_all_cost = time_end - time_start # 計算總耗時(抽樣時間+計算時間)
            print(f"time all cost(s): {time_all_cost}s")
            time_total += time_all_cost
            print("LOSS_LIMIT=", LOSS_LIMIT)
        else:
            print(f"優化失敗: {result.message}")
    if ROUND != 1:
        if loss_total/ROUND < LOSS_LIMIT[EXPLAIN_DATA]:
            countAll += 1
        if LOOPNUM > 1:
            avgAll += loss_total/ROUND
        
        print(f"此為ID{ID[DATASET]}資料集, 解釋第{EXPLAIN_DATA}筆資料, mode{MODE}, 抽樣{SAMPLING_NUM[DATASET]}個, 總做了{ROUND}次")
        print(f"平均抽樣時間(s): {sampling_time_total/ROUND}s")
        print(f"平均時間(s): {time_total/ROUND}s")
        print(f"平均差距: {loss_total/ROUND}")
        print(f"最大差距: {loss_max}")
        print(f"最小差距: {loss_min}")
        print(f"小於{LOSS_LIMIT[EXPLAIN_DATA]}的次數: {count}")
        
        np.savetxt(f"{LOCATION}\\AllLossList_mode{MODE}_exd{EXPLAIN_DATA}_round{ROUND}.txt", allLossList)
        np.savetxt(f"{LOCATION}\\AllList_mode{MODE}_exd{EXPLAIN_DATA}_round{ROUND}.txt", allSampList)
        np.savetxt(f"{LOCATION}\\SpaceList_mode{MODE}_exd{EXPLAIN_DATA}_round{ROUND}.txt", allSpacList)
        np.savetxt(f"{LOCATION}\\AllShapValueList_mode{MODE}_exd{EXPLAIN_DATA}_round{ROUND}.txt", allShapValue)

if __name__=='__main__':
    LOOPNUM = 50 # 解釋資料數量
    DATASET = 0 # 選擇資料集
    ID = [186, 519, 563, 1, 165, 60, 544]
    EXPLAIN_DATA = 0 # 選擇要解釋第幾筆資料(單筆解釋)
    MODE = 5 # 隨機方法:0, 隨機配對抽樣:1, Sobol:2, Halton:3, 凸型費氏:4, 低差異費氏配對:5
    COMP_MODE = 4
    # 隨機選取特徵子集的數量: 32, 34, 36, 22, 22, 14, 32(mode4)
    SAMPLING_NUM = [32, 34, 36, 22, 22, 14, 50, 32]
    ROUND = 50 # 要計算幾次
    GOLDEN_RATIO = (5**0.5 - 1)/2
    LOCATION = f"SHAPSampling\\result_data\\{ID[DATASET]}"
    LOSS_LIMIT = dict()

    reCalcu = False #是否重新計算ANS_LIST
    ansPath = f"{LOCATION}\\ANS"
    lossPath = f"{LOCATION}\\LOSS"
    fibonacciSeq = {0:0, 1:1}

    countAll = 0
    avgAll = 0

    X_train, X_test, y_train, y_test = _setData()
    # Number of features(M)
    columns = X_train.columns.tolist()
    featureNum = len(columns)

    model = Model()

    if SAMPLING_NUM[DATASET] >= 2**featureNum: SAMPLING_NUM[DATASET] = 2**featureNum-1
    if LOOPNUM < 1 : LOOPNUM = 1
    for _ in range(LOOPNUM):
        binToAnsDict = {} # 紀錄已計算的預測結果
        # predict data
        X_predictData = X_test.iloc[[EXPLAIN_DATA]]
        dtypeDict = X_train.dtypes.apply(lambda x: x.name).to_dict()
        midData = pd.DataFrame([X_test.median()])
        midData = midData.astype(dtypeDict)
        ansPredict, midPredict = model.getAnsAndMidPredict()

        if not os.path.exists(ansPath): os.makedirs(ansPath)
        if not os.path.exists(lossPath): os.makedirs(lossPath)

        if not os.path.exists(ansPath + f"\\ans_{EXPLAIN_DATA}.txt") or reCalcu:
            ANS_LIST = _getExactShapValue(model.model)
        else: ANS_LIST = np.loadtxt(ansPath + f"\\ans_{EXPLAIN_DATA}.txt") # 全包含的SHAP值(精準SHAP值)

        if not os.path.exists(lossPath + f"\\loss_mode{COMP_MODE}.npy"): 
            samplingList = sampling("COMP_MODE")
            LOSS_LIMIT = getLOSS()
        else:
            with open(lossPath + f"\\loss_mode{COMP_MODE}.npy", 'rb') as file:
                LOSS_LIMIT = np.load(file, allow_pickle=True).item() # 字典[EXPLAIN_DATA] 保存上限設定值(mode4)
            if LOSS_LIMIT.get(EXPLAIN_DATA, -1) <= 0:
                samplingList = sampling("COMP_MODE")
                LOSS_LIMIT = getLOSS()

        print("ANS_LIST=",ANS_LIST)
        print("LOSS_LIMIT=",LOSS_LIMIT) 

        if MODE == 4: ROUND = 1
        mainFunc()
        EXPLAIN_DATA += 1
        
    print("countAll =",countAll)
    print("avgAll =", avgAll/LOOPNUM)