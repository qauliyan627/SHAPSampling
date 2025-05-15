import math
import random
import time
import json

from scipy.optimize import minimize
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
import xgboost as xgb

# import dataset
uciDataset = fetch_ucirepo(id=186) 
# Feature Engineering
X = uciDataset.data.features
y = uciDataset.data.targets
# Number of features(M)
columns = X.columns
featureNum = len(columns)
print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, ## predictors only
                                                    y,
                                                    test_size=0.20, 
                                                    random_state=0)
#print(X_test)

# Modelling
# Train model
model = xgb.XGBRegressor(objective="reg:squarederror")
model.fit(X_train, y_train)
print('model ok')

# predict
def predict(data):
    prediction_pandas = model.predict(data)[0]
    #print(f"predict: {prediction_pandas}")
    return prediction_pandas
# predict data
X_predictData = X_test.iloc[[0]]
print(X_predictData)
midData = pd.DataFrame([X_test.mean()])
ansPredict = predict(X_predictData)
midPredict = predict(midData)

# transiton: h()
def transFun(z):
    transData = X_predictData.copy(deep=True)
    for i in range(len(z)):
        if z[i] == '0':
            transData.loc[:, columns[i]] = midData.loc[0, columns[i]]
    return transData

# Kernel Weight: Pi_x()
def weightFunc(subsetSize):
    weightNum = (featureNum-1) / (math.comb(featureNum,subsetSize)*subsetSize*(featureNum-subsetSize))
    #print(weightNum)
    return weightNum

binToAnsDict = {}
def objective_function(variables):
    varDict = {}
    for i in range(featureNum):
        varDict[f"x{i}"] = variables[i]
    total_sum = 0
    
    for i in range(1,2**featureNum-1):
        if i not in samplingList:
            if SAMPLING_NUM == "max":
                pass
            else:
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
            predictAns = predict(transData)
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
    if samplingNum == 'max': return range(1, 2**featureNum-1)
    samplingList = []
    while True:
        r = random.randint(1, 2**featureNum-1)
        if r in samplingList:
            continue
        else:
            samplingList.append(r)
        if len(samplingList) >= samplingNum:
            break
    print(f"randomList: {sorted(samplingList)}")
    return samplingList

def FibSampling(samplingNum):
    fibCount = 2
    fibList = [1, 2]
    for k in range(1, 2**featureNum-1):
        fibNum = fibList[k]+fibList[k-1]
        if fibNum > 2**featureNum-1:
            break
        fibList.append(fibNum)
        fibCount+=1
    # 反向抽取
    tempList = fibList
    for k in tempList:
        if 2**featureNum - k not in fibList:
            fibList.append(2**featureNum - k)
    print(f"fibList: {sorted(fibList)}")
    return fibList

GOLDEN_RATIO = 0.61803398875
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
    print(f"GoldenSampling: {sorted(samplingList)}")
    return samplingList

def aveFibSampling(samp):
    FIB_LIST = [0,1,2,3,5,8,13,21,33,54]
    passList = [1,9,30]
    coverNumList =[]
    count = 0
    last = -1
    tempList = []
    intervalSize = 2**featureNum//samp
    for _ in range(len(passList)): 
        while True:
            temp = random.randint(1,SAMPLING_NUM)
            if temp in coverNumList or temp in passList: continue
            coverNumList.append(temp)
            break
    coverNumList = [8,14,31]
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
    print(f"aveFibList: {tempList}")
    return tempList

def pairedSampling(samp):
    tempList = []
    for i in range(2**featureNum//2, 2**featureNum):
        pass

# 全包含的SHAP值(精準SHAP值)
ANS_LIST = [-0.10743713601999705, 0.4198516820913887, -0.1871362529799483, -0.04834011330128862, -0.04659699356051017, -0.30819252728339197, -0.09294528711643413, -0.0030462765588805674, -0.12074284239519217, 0.00002968639930678, -0.37836063773085904]
LOCATION = "plot_data\\"
ROUND = 100 # 要計算幾次
MODE = 1 # 隨機方法0, 傳統費式1, 黃金抽樣2, 平均費式3, 凸型費式4
GAP_LIMIT = 0.5 # 保存上限設定值
SAMPLING_NUM = 32 # 隨機選取特徵子集的數量32

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
if SAMPLING_NUM == "max": ROUND=1
for j in range(ROUND):
    print(f"j={j}")
    
    # samplingList: 特徵子集抽樣 array = 1~2**featureNum-1
    print(f"SAMPLING_NUM = {SAMPLING_NUM}")
    
    time_start = time.time() # 開始計算時間
    
    if MODE == 0 or SAMPLING_NUM == "max":
        samplingList = randomSampling(SAMPLING_NUM)
    elif MODE == 1:
        samplingList = FibSampling(SAMPLING_NUM)
    elif MODE == 2:
        samplingList = GoldenSampling(SAMPLING_NUM)
    elif MODE == 3:
        samplingList = aveFibSampling(SAMPLING_NUM)
    
    time_end = time.time() # 抽樣結束時間
    samplingTime = time_end - time_start # 計算抽樣時間
    print(f"samplingTime: {samplingTime}")
    
    samplingList = sorted(samplingList) # 排序抽樣結果
    print(f"len: {len(samplingList)}")
    
    initial_guess = np.arange(featureNum)
    constraints = ({'type': 'eq', 'fun': equality_constraint})
    options = {'maxiter': 10000}
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
        
        gap = getGap(optimal_variables) # 取得和精準SHAP值之間的差距
        if gap < GAP_LIMIT: count+=1 # 計算差距小於設定值的次數
        if gap > gap_max: gap_max = gap
        if gap < gap_min: gap_min = gap
        if gap < GAP_LIMIT: 
            gapSampList.append(samplingList)
            gapSpacList.append(getSpac(samplingList))
            gapList.append(gap)
        allGapList.append(gap)
        gap_total += gap
        print(f"差距: {gap}")
        time_all_cost = time_end - time_start # 計算總耗時(抽樣時間+計算時間)
        print(f"time all cost(s): {time_all_cost}s")
        print(f"sampling cost(s): {samplingTime}s")
        time_total += time_all_cost
        sampling_time_total += samplingTime
    else:
        print(f"優化失敗: {result.message}")
    if MODE == 3:
        break
if not MODE == 3:
    print(f"此為mode{MODE}, 總做了{ROUND}次")
    print(f"平均抽樣時間(s): {sampling_time_total/ROUND}s")
    print(f"平均時間(s): {time_total/ROUND}s")
    print(f"平均差距: {gap_total/ROUND}")
    print(f"最大差距: {gap_max}")
    print(f"最小差距: {gap_min}")
    print(f"小於{GAP_LIMIT}的次數: {count}")
    print(f"小於{GAP_LIMIT}的抽選: {gapSampList}")
    # print(f"allSpacList: {allSpacList}")
    
    if len(gapSampList) > 0:
        np.savetxt(f"{LOCATION}GapSampList_mode{MODE}_round{ROUND}.txt", gapSampList)
        np.savetxt(f"{LOCATION}GapSpacList_mode{MODE}_round{ROUND}.txt", gapSpacList)
        saveGapSampList(gapSampList)
    np.savetxt(f"{LOCATION}AllGapList_mode{MODE}_round{ROUND}.txt", allGapList)
    np.savetxt(f"{LOCATION}AllList_mode{MODE}_round{ROUND}.txt", allSampList)
    np.savetxt(f"{LOCATION}SpaceList_mode{MODE}_round{ROUND}.txt", allSpacList)
    np.savetxt(f"{LOCATION}AllShapValueList_mode{MODE}_round{ROUND}.txt", allShapValue)