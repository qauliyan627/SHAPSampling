import os
import math
import random
import time

from scipy.optimize import minimize
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
import xgboost as xgb

def startSet(): # import dataset
    uciDataset = fetch_ucirepo(id=ID[DATASET])
    # Feature Engineering
    X = uciDataset.data.features
    y = uciDataset.data.targets
    
    if ID[DATASET] == 1:
        X['Sex'].replace(['M', 'F', 'I'], [0, 1, 2], inplace=True)
        print(X)
    elif ID[DATASET] == 544:
        cat_columns = X.select_dtypes(['object']).columns
        for cc in cat_columns:
            codes, uniques = pd.factorize(X[cc])
            X[cc] = codes
        cat_columns = y.select_dtypes(['object']).columns
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

class Model():
    def __init__(self):
        # Modelling
        # Train model
        self.model = xgb.XGBRegressor(objective="reg:squarederror")
        self.model.fit(X_train, y_train)
        print('model ok')
        
    def predictDataSet(self):
        ansPredict = self.predict(X_predictData)
        midPredict = self.predict(midData)
        return ansPredict, midPredict

    # predict
    def predict(self, data):
        prediction_pandas = self.model.predict(data)[0]
        #print(f"predict: {prediction_pandas}")
        return prediction_pandas   

def transFun(z): # transiton: h()
    transData = X_predictData.copy(deep=True)
    for i in range(len(z)):
        if z[i] == '0':
            transData.loc[:, columns[i]] = midData.loc[0, columns[i]]
    return transData

def weightFunc(subsetSize): # Kernel Weight: Pi_x()
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
        transData = transFun(i_bin)
        tempTerm = midPredict
        for k in range(featureNum):
            if i_bin[k] == '1':
                tempTerm += varDict[f"x{k}"]
        if i_bin in binToAnsDict.keys():
            predictAns = binToAnsDict[i_bin]
        else:
            predictAns = model.predict(transData)
            binToAnsDict[i_bin] = predictAns
        term = ((predictAns - (tempTerm))**2)*weightFunc(i_bin.count('1'))
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

def getGap(optimal_variables):
    gap = 0
    for i in range(featureNum):
        gap += abs(ANS_LIST[i] - optimal_variables[i])
    print("gap in getGap:",gap)
    return gap

def getSpac(spList):
    spacList = []
    for i in range(1,len(spList)):
        spacList.append(abs(spList[i] - spList[i-1]))
    return spacList

def saveGapSampList(sampList):
    with open(f"{LOCATION}ALLsamplingGapList.txt", 'a') as f:
        np.savetxt(f,sampList)
    f.close()

def randomSampling(samplingNum):
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

def FibSampling(samplingNum):# 凹型抽樣
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

def GoldenSampling(samplingNum):
    samplingList = []
    last = random.randint(1, 2**featureNum-1)
    samplingList.append(last)
    while True:
        next = math.fmod((last - 1) / (2**featureNum-1) + GOLDEN_RATIO, 1.0)
        last = math.floor(next * (2**featureNum-1)) + 1
        if last not in samplingList: samplingList.append(last)
        if len(samplingList) >= samplingNum:
            break
    return samplingList

def aveFibSampling(samp):
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

def pairedSampling(): # 凸型配對(左右對稱)
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

def stratifiedSampling():
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

def fibonacci(n):
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

def sampling(sampling_num, mode=0):
    time_start = time.time() # 開始計算時間
    if sampling_num == "COMP_MODE": mode = COMP_MODE
    if sampling_num == "max":
        samplingList = list(range(1, 2**featureNum-1))
    elif mode == 0: 
        samplingList = randomSampling(sampling_num)
    elif mode == 1:
        samplingList = FibSampling(sampling_num)
    elif mode == 2:
        samplingList = GoldenSampling(sampling_num)
    elif mode == 3:
        samplingList = aveFibSampling(sampling_num)
    elif mode == 4:
        samplingList = pairedSampling()
    elif mode == 5:
        samplingList = stratifiedSampling()
    time_end = time.time() # 抽樣結束時間
    samplingTime = time_end - time_start # 計算抽樣時間
    samplingList.sort()
    print(f"samplingTime={samplingTime}")
    print(f"samplingList={samplingList}")
    print(f"len: {len(samplingList)}")
    return samplingList
    
def getANSandGAP(sampling_num): # 計算ANS_LIST, 計算GAP_LIMIT(COMP_MODE)
    initial_guess = np.arange(featureNum)
    constraints = ({'type': 'eq', 'fun': equality_constraint})
    options = {'maxiter': 10000}
    time_start = time.time()
    result = minimize(objective_function, initial_guess, constraints=constraints, method='SLSQP') # 計算SHAP值
    time_end = time.time() # SHAP值計算結束時間
    print(f"計算時間={time_end - time_start}")
    if result.success:
        optimal_variables = result.x
        if sampling_num == "max":
            np.savetxt(f"{LOCATION}\\ANS\\ans_{EXPLAIN_DATA}.txt", optimal_variables)
            return np.loadtxt(f"{LOCATION}\\ANS\\ans_{EXPLAIN_DATA}.txt")

        gap = getGap(optimal_variables) # 取得和精準SHAP值之間的差距
        if sampling_num == "COMP_MODE": # 沒有檔案，保存GAP_LIMIT
            print("= "*10)
            GAP_LIMIT[EXPLAIN_DATA] = gap
            np.save(f"{LOCATION}\\GAP\\gap_mode{COMP_MODE}.npy", GAP_LIMIT)
            return np.load(f"{LOCATION}\\GAP\\gap_mode{COMP_MODE}.npy", allow_pickle=True).item()
    else:
        print(f"優化失敗: {result.message}")

DATASET = 4 # 選擇資料集
ID = [186, 519, 563, 1, 165, 60, 544]
EXPLAIN_DATA = 1 # 選擇要解釋第幾筆資料(單筆解釋)
MODE = 2 # 隨機方法0, 傳統費氏(凹型)1, 黃金抽樣(低序列差異)2, 平均費氏3, 對稱費氏(凸型)4, 分層費氏5
COMP_MODE = 4
# 隨機選取特徵子集的數量: 32, 34, 36, 22, 22, 14(mode4)
SAMPLING_NUM = [32, 34, 36, 22, 22, 14, 50]
ROUND = 100 # 要計算幾次
GOLDEN_RATIO = 0.61803398875
LOCATION = f"SHAPSampling\\plot_data\\{ID[DATASET]}"
GAP_LIMIT = dict()

samplingList = []
binToAnsDict = {} # 紀錄已計算的預測結果
time_total = 0
sampling_time_total = 0
gap_total = 0
gap_max = 0
gap_min = 9999
count = 0
gapList = [] # 小於GAP_LIMIT的子級組
gapSampList = []
gapSpacList = []
allGapList = []
allSampList = []
allSpacList = [] # 抽選子集組的各子集距離
allShapValue = [] # 記錄每次計算的SHAP值
fibonacciSeq = {0:0, 1:1}

X_train, X_test, y_train, y_test = startSet()
dtypeDict = X_train.dtypes.apply(lambda x: x.name).to_dict()
# Number of features(M)
columns = X_train.columns.tolist()
featureNum = len(columns)
if SAMPLING_NUM[DATASET] >= 2**featureNum: SAMPLING_NUM[DATASET] = 2**featureNum-1
# predict data
X_predictData = X_test.iloc[[EXPLAIN_DATA]]
midData = pd.DataFrame([X_test.median()])
midData = midData.astype(dtypeDict)

model = Model()
ansPredict, midPredict = model.predictDataSet()

reCalcu = False #是否重新計算ANS_LIST
if not os.path.exists(f"{LOCATION}\\ANS\\ans_{EXPLAIN_DATA}.txt") or reCalcu:
    samplingList = sampling("max")
    ANS_LIST = getANSandGAP("max")
else: ANS_LIST = np.loadtxt(f"{LOCATION}\\ANS\\ans_{EXPLAIN_DATA}.txt") # 全包含的SHAP值(精準SHAP值)
if not os.path.exists(f"{LOCATION}\\GAP\\gap_mode{COMP_MODE}.npy"): 
    samplingList = sampling("COMP_MODE")
    GAP_LIMIT = getANSandGAP("COMP_MODE")
else:
    with open(f"{LOCATION}\\GAP\\gap_mode{COMP_MODE}.npy", 'rb') as file:
        GAP_LIMIT = np.load(file, allow_pickle=True).item() # 字典[EXPLAIN_DATA] 保存上限設定值(mode4)
    if GAP_LIMIT.get(EXPLAIN_DATA, -1) <= 0:
        samplingList = sampling("COMP_MODE")
        GAP_LIMIT = getANSandGAP("COMP_MODE")
print("ANS_LIST=",ANS_LIST)
print("GAP_LIMIT=",GAP_LIMIT)

for j in range(ROUND):
    print(f"j={j}")
    
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
        if SAMPLING_NUM[DATASET] == "max" and len(ANS_LIST) == 0:
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
        
        gap = getGap(optimal_variables) # 取得和精準SHAP值之間的差距
        if MODE == COMP_MODE and GAP_LIMIT.get(EXPLAIN_DATA, -1) == -1:
            GAP_LIMIT[EXPLAIN_DATA] = gap
            np.save(f"{LOCATION}\\GAP\\gap_mode{COMP_MODE}.npy", GAP_LIMIT)
            GAP_LIMIT = np.load(f"{LOCATION}\\GAP\\gap_mode{COMP_MODE}.npy", allow_pickle=True).item()
        if gap < GAP_LIMIT[EXPLAIN_DATA]: count+=1 # 計算差距小於設定值的次數
        if gap > gap_max: gap_max = gap
        if gap < gap_min: gap_min = gap
        if gap < GAP_LIMIT[EXPLAIN_DATA]: 
            gapSampList.append(samplingList)
            gapSpacList.append(getSpac(samplingList))
            gapList.append(gap)
        allGapList.append(gap)
        gap_total += gap
        print(f"差距: {gap}")
        time_all_cost = time_end - time_start # 計算總耗時(抽樣時間+計算時間)
        print(f"time all cost(s): {time_all_cost}s")
        time_total += time_all_cost
        print("GAP_LIMIT=", GAP_LIMIT)
    else:
        print(f"優化失敗: {result.message}")
    if MODE == 1 or MODE == 4:
        break
if not MODE == 1 and not MODE == 4:
    print(f"此為ID{ID[DATASET]}資料集, 解釋第{EXPLAIN_DATA}筆資料, mode{MODE}, 抽樣{SAMPLING_NUM[DATASET]}個, 總做了{ROUND}次")
    print(f"平均抽樣時間(s): {sampling_time_total/ROUND}s")
    print(f"平均時間(s): {time_total/ROUND}s")
    print(f"平均差距: {gap_total/ROUND}")
    print(f"最大差距: {gap_max}")
    print(f"最小差距: {gap_min}")
    print(f"小於{GAP_LIMIT[EXPLAIN_DATA]}的次數: {count}")
    print(f"小於{GAP_LIMIT[EXPLAIN_DATA]}的抽選: {gapSampList}")
    # print(f"allSpacList: {allSpacList}")
    
    if len(gapSampList) > 0:
        np.savetxt(f"{LOCATION}\\GapSampList_mode{MODE}_round{ROUND}.txt", gapSampList)
        np.savetxt(f"{LOCATION}\\GapSpacList_mode{MODE}_round{ROUND}.txt", gapSpacList)
        #saveGapSampList(gapSampList)
    np.savetxt(f"{LOCATION}\\AllGapList_mode{MODE}_round{ROUND}.txt", allGapList)
    np.savetxt(f"{LOCATION}\\AllList_mode{MODE}_round{ROUND}.txt", allSampList)
    np.savetxt(f"{LOCATION}\\SpaceList_mode{MODE}_round{ROUND}.txt", allSpacList)
    np.savetxt(f"{LOCATION}\\AllShapValueList_mode{MODE}_round{ROUND}.txt", allShapValue)
    
    