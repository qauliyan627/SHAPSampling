import os
import math
import random
import time
import logging

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
    varDict = {}
    for i in range(featureNum):
        varDict[f"x{i}"] = variables[i]
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
                tempTerm += varDict[f"x{k}"]
        if i_bin in binToAnsDict.keys():
            predictAns = binToAnsDict[i_bin]
        else:
            predictAns = model.predict(transData)
            binToAnsDict[i_bin] = predictAns
        term = ((predictAns - (tempTerm))**2)*_Pi_x(i_bin.count('1'))
        total_sum += term
    return total_sum

def equality_constraint(variables):
    varDict = {}
    for i in range(featureNum):
        varDict[f"x{i}"] = variables[i]
    equaTerm = midPredict
    for i in range(featureNum):
        equaTerm += varDict[f"x{i}"]
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
    samplingList = []
    while True:
        r = random.randint(1, 2**featureNum-1)
        if r in samplingList:
            continue
        else:
            samplingList.append(r)
        if len(samplingList) >= samplingNum:
            break
    return samplingList

def FibSampling(): # mode1: 使用費氏數列的凹型抽樣
    i = 2
    samplingList = []
    while True:
        temp = fibonacci(i)
        if temp >= 2**featureNum: break
        samplingList.append(temp)
        i+=1
    # 反向費氏
    maxSampNum = 2**featureNum-1
    if maxSampNum not in samplingList: samplingList.append(maxSampNum)
    i = 2
    while True:
        temp = fibonacci(i)
        if temp >= 2**featureNum: break
        if maxSampNum - temp not in samplingList: samplingList.append(maxSampNum - temp)
        i+=1
    return samplingList

def GoldenSampling(samplingNum): # mode2: 使用黃金比例的低差異序列抽樣
    totalSetNum = 2**featureNum-1
    count = 0
    samplingList = []
    index = random.randint(0, totalSetNum)
    index = math.fmod(GOLDEN_RATIO * index, 1.0)
    i = int(index * (totalSetNum - 1) + 1)
    samplingList.append(i)
    last = index
    for _ in range(samplingNum-1):
        while True:
            last = math.fmod(last + GOLDEN_RATIO, 1.0)
            i = int(last * (totalSetNum - 1) + 1)
            if i in samplingList:
                count += 1
                print("continue")
                continue
            samplingList.append(i)
            #print(samplingList)
            break
    
    print("continue c:", count)
    return samplingList

def aveFibSampling(samp): # mode3: 分割區間使用費氏數列
    FIB_LIST = [0,1,2,3,5,8,13,21,33,54,]
    passList = []
    coverNumList =[]
    count = 0
    last = -1
    tempList = []
    intervalSize = 2**featureNum//samp
    for _ in range(len(passList)): 
        while True:
            temp = random.randint(1,SAMPLING_NUM[DATASET])
            if temp in coverNumList or temp in passList: continue
            coverNumList.append(temp)
            break
    coverNumList = []
    for i in range(1, 2**featureNum, intervalSize):
        ctu = True
        count+=1
        if count in passList:
            continue
        while True:
            ran = random.randint(0,len(FIB_LIST)-1)
            if ran == last:
                continue
            else:
                last = ran
                tempList.append(i+FIB_LIST[ran])
                if count in coverNumList and ctu:
                    ctu = False
                    continue
                break
    return tempList

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

def stratifiedSampling(): # mode5: 費氏數列 + 分層抽樣
    stratDict = {}
    for i in range(1, 2**featureNum): # 製作分層Dict
        i_bin = format(i, 'b')
        i_bin = i_bin.zfill(featureNum)
        oneCount = i_bin.count('1')
        tempList = stratDict.get(oneCount, [])
        tempList.append(i)
        stratDict[oneCount] = tempList
    # 取得個曾要抽的比例
    sampRatio = []
    total = 0
    for i in range(1, featureNum):
        sampRatio.append(len(stratDict[i]))
        total+=sampRatio[i-1]
    sampRatio = [x/total for x in sampRatio]
    sampNum = [round(x*SAMPLING_NUM[DATASET]) for x in sampRatio]
    # 開始抽樣
    sampList = []
    for i in range(1, featureNum):
        maxFibNum = 0 # 最大可抽費數 F(n)
        j = 2
        while True:
            if fibonacci(j) <= len(stratDict[i]):
                j+=1
            else:
                j-=1
                break
        maxFibNum = j
        for j in range(sampNum[i-1]):
            while True:
                temp = random.randint(2, maxFibNum)
                if stratDict[i][temp] in sampList:
                    continue
                else:
                    sampList.append(stratDict[i][temp])
                    break
    return sampList

def ldFibSampling(samp): # mode6: 費氏數列 + 低差異序列想法(挑選最大區間抽樣)
    top = 2**featureNum//2-1
    but = 2**featureNum
    tempList = [top, but]
    n_top = top
    n_but = but
    for _ in range(samp//2):
        # 計算最大可用費氏數
        ran = n_but - n_top - 1
        maxFib = 0
        while True:
            if ran < fibonacci(maxFib):
                maxFib -= 1
                break
            else: maxFib += 1
        # 抽樣
        count=0
        while True:
            count += 1
            if count > 1000: ranFib = random.randint(1, ran)
            else: ranFib = fibonacci(random.randint(1, maxFib))
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
        samplingList = FibSampling()
    elif mode == 2:
        samplingList = GoldenSampling(sampling_num)
    elif mode == 3:
        samplingList = aveFibSampling(sampling_num)
    elif mode == 4:
        samplingList = pairedSampling()
    elif mode == 5:
        samplingList = stratifiedSampling()
    elif mode == 6:
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
        return np.load(f"{LOCATION}\\LOSS\\loss_mode{COMP_MODE}.npy", allow_pickle=True).item()
    else:
        print(f"優化失敗: {result.message}")

DATASET = 0 # 選擇資料集
ID = [186, 519, 563, 1, 165, 60, 544]
EXPLAIN_DATA = 0 # 選擇要解釋第幾筆資料(單筆解釋)
MODE = 6 # 隨機方法0, 傳統費氏(凹型)1, 黃金抽樣(低序列差異)2, 平均費氏3, 對稱費氏(凸型)4, 分層費氏5
COMP_MODE = 4
# 選取特徵子集的數量
SAMPLING_NUM = 10
ROUND = 50 # 要計算幾次
GOLDEN_RATIO = (5**0.5 - 1)/2
LOCATION = f"SHAPSampling\\plot_data\\{ID[DATASET]}"
LOSS_LIMIT = dict()

maxLoss = 0
maxLossSampNum = 0
minLoss = 10000
minLossSampNum = 0
all_SAMPLING_NUM_gapList = {}
sampAdd = 20
while True:
    samplingList = []
    binToAnsDict = {} # 紀錄已計算的預測結果
    time_total = 0
    sampling_time_total = 0
    gap_total = 0
    gap_max = 0
    gap_min = 9999
    count = 0
    gapList = [] # 小於LOSS_LIMIT的子級組
    gapSampList = []
    gapSpacList = []
    allLossList = []
    allSampList = []
    allSpacList = [] # 抽選子集組的各子集距離
    allShapValue = [] # 記錄每次計算的SHAP值
    fibonacciSeq = {0:0, 1:1}

    X_train, X_test, y_train, y_test = _setData()
    dtypeDict = X_train.dtypes.apply(lambda x: x.name).to_dict()
    # Number of features(M)
    columns = X_train.columns.tolist()
    featureNum = len(columns)
    if SAMPLING_NUM >= 2**featureNum: SAMPLING_NUM = 2**featureNum-1
    # predict data
    X_predictData = X_test.iloc[[EXPLAIN_DATA]]
    midData = pd.DataFrame([X_test.median()])
    midData = midData.astype(dtypeDict)

    model = Model()
    ansPredict, midPredict = model.getAnsAndMidPredict()

    reCalcu = False #是否重新計算ANS_LIST
    ansPath = f"{LOCATION}\\ANS"
    if not os.path.exists(ansPath): os.makedirs(ansPath)
    if not os.path.exists(ansPath + f"\\ans_{EXPLAIN_DATA}.txt") or reCalcu:
        ANS_LIST = _getExactShapValue(model.model)
    else: ANS_LIST = np.loadtxt(ansPath + f"\\ans_{EXPLAIN_DATA}.txt") # 全包含的SHAP值(精準SHAP值)
    gapPath = f"{LOCATION}\\LOSS"
    if not os.path.exists(gapPath): os.makedirs(gapPath)
    if not os.path.exists(gapPath + f"\\gap_mode{COMP_MODE}.npy"): 
        samplingList = sampling("COMP_MODE")
        LOSS_LIMIT = getLOSS()
    else:
        with open(gapPath + f"\\gap_mode{COMP_MODE}.npy", 'rb') as file:
            LOSS_LIMIT = np.load(file, allow_pickle=True).item() # 字典[EXPLAIN_DATA] 保存上限設定值(mode4)
        if LOSS_LIMIT.get(EXPLAIN_DATA, -1) <= 0:
            samplingList = sampling("COMP_MODE")
            LOSS_LIMIT = getLOSS()
    print("ANS_LIST=",ANS_LIST)
    print("LOSS_LIMIT=",LOSS_LIMIT)

    for j in range(ROUND):
        print(f"j={j}")
        
        # samplingList: 特徵子集抽樣 array = 1~2**featureNum-1
        print(f"SAMPLING_NUM = {SAMPLING_NUM}")
        samplingList = sampling(SAMPLING_NUM, MODE)
        
        initial_guess = np.arange(featureNum)
        constraints = ({'type': 'eq', 'fun': equality_constraint})
        options = {'maxiter': 10000}
        time_start = time.time()
        result = minimize(objective_function, initial_guess, constraints=constraints, method='SLSQP') # 計算SHAP值
        time_end = time.time() # SHAP值計算結束時間
        if result.success:
            minimum_value = result.fun
            optimal_variables = result.x
            if SAMPLING_NUM == "max" and len(ANS_LIST) == 0:
                np.savetxt(f"{LOCATION}\\ANS\\ans_{EXPLAIN_DATA}.txt", optimal_variables)
                ANS_LIST = np.loadtxt(f"{LOCATION}\\ANS\\ans_{EXPLAIN_DATA}.txt")
            
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
            
            gap = getLoss(optimal_variables) # 取得和精準SHAP值之間的差距
            if MODE == COMP_MODE and LOSS_LIMIT.get(EXPLAIN_DATA, -1) == -1:
                LOSS_LIMIT[EXPLAIN_DATA] = gap
                np.save(f"{LOCATION}\\LOSS\\gap_mode{COMP_MODE}.npy", LOSS_LIMIT)
                LOSS_LIMIT = np.load(f"{LOCATION}\\LOSS\\gap_mode{COMP_MODE}.npy", allow_pickle=True).item()
            if gap < LOSS_LIMIT[EXPLAIN_DATA]: count+=1 # 計算差距小於設定值的次數
            if gap > gap_max: gap_max = gap
            if gap < gap_min: gap_min = gap
            if gap < LOSS_LIMIT[EXPLAIN_DATA]: 
                gapSampList.append(samplingList)
                gapSpacList.append(getSpac(samplingList))
                gapList.append(gap)
            allLossList.append(gap)
            gap_total += gap
            print(f"差距: {gap}")
            time_all_cost = time_end - time_start # 計算總耗時(抽樣時間+計算時間)
            print(f"time all cost(s): {time_all_cost}s")
            time_total += time_all_cost
            print("LOSS_LIMIT=", LOSS_LIMIT)
        else:
            print(f"優化失敗: {result.message}")
        if MODE == 1 or MODE == 4:
            break
    if not MODE == 1 and not MODE == 4:
        print(f"此為ID{ID[DATASET]}資料集, 解釋第{EXPLAIN_DATA}筆資料, mode{MODE}, 抽樣{SAMPLING_NUM}個, 總做了{ROUND}次")
        print(f"平均抽樣時間(s): {sampling_time_total/ROUND}s")
        print(f"平均時間(s): {time_total/ROUND}s")
        print(f"平均差距: {gap_total/ROUND}")
        print(f"最大差距: {gap_max}")
        print(f"最小差距: {gap_min}")
        print(f"小於{LOSS_LIMIT[EXPLAIN_DATA]}的次數: {count}")
        print(f"小於{LOSS_LIMIT[EXPLAIN_DATA]}的抽選: {gapSampList}")
        # print(f"allSpacList: {allSpacList}")
        
        if len(gapSampList) > 0:
            np.savetxt(f"{LOCATION}\\LossSampList_mode{MODE}_round{ROUND}.txt", gapSampList)
            np.savetxt(f"{LOCATION}\\LossSpacList_mode{MODE}_round{ROUND}.txt", gapSpacList)
            #saveLossSampList(gapSampList)
        np.savetxt(f"{LOCATION}\\AllLossList_mode{MODE}_round{ROUND}.txt", allLossList)
        np.savetxt(f"{LOCATION}\\AllList_mode{MODE}_round{ROUND}.txt", allSampList)
        np.savetxt(f"{LOCATION}\\SpaceList_mode{MODE}_round{ROUND}.txt", allSpacList)
        np.savetxt(f"{LOCATION}\\AllShapValueList_mode{MODE}_round{ROUND}.txt", allShapValue)
        
    if gap_total/ROUND > maxLoss: 
        maxLoss = gap_total/ROUND
        maxLossSampNum = SAMPLING_NUM
    if gap_total/ROUND < minLoss: 
        minLoss = gap_total/ROUND
        minLossSampNum = SAMPLING_NUM
    all_SAMPLING_NUM_gapList[len(samplingList)] = gap_total/ROUND
    SAMPLING_NUM *= 2
    if SAMPLING_NUM >= 650: break
print(f"all_SAMPLING_NUM_gapList = {all_SAMPLING_NUM_gapList}")
print(f"maxLoss = {maxLoss}, maxLossSampNum = {maxLossSampNum}")
print(f"minLoss = {minLoss}, minLossSampNum = {minLossSampNum}")