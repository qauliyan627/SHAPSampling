import os
import math
import random
import datetime
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
    X = pd.read_csv(DATASET_LOC + "X.csv")
    y = pd.read_csv(DATASET_LOC + "y.csv")
    
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
    np.savetxt(f"{ansPath}\\ans_{EXPLAIN_DATA}.txt", shap_values[0].values)
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

def objective_matrix(param, X_mat, y_vec, W_mat):
    params = [midPredict]
    # params 現在是 [a, b, midPredict]
    for i in range(featureNum):
        params.append(param[i])
    beta = np.array(params).reshape(-1, 1) # 將 params 轉換為 (2, 1) 的列向量
    # np.insert(beta, -1, midPredict)
    # 計算預測值向量 y_hat = X * beta
    y_hat = X_mat @ beta # 矩陣乘法

    # 計算殘差向量 e = y - y_hat
    errors = y_vec.reshape(-1, 1) - y_hat # 確保 y_vec 也是列向量進行廣播或明確 reshape

    # 計算加權平方誤差：errors.T @ W @ errors
    # np.dot(errors.T, W_mat @ errors) 也可以
    weighted_squared_error = errors.T @ W_mat @ errors
    
    # minimize 函數期望一個標量返回值，所以我們取結果的第一個元素
    return weighted_squared_error[0, 0]

def equality_constraint(params):
    equaTerm = midPredict
    for i in range(featureNum):
        equaTerm += params[i]
    equaTerm -= ansPredict
    return equaTerm  # 所有shapley value相加等於預測結果

def toBinList(tenList): #將list轉換為二進制
    tempList = []
    for i in tenList:
        i_bin = format(i, 'b')
        i_bin = i_bin.zfill(featureNum)
        i_bin = list(i_bin)
        tempList.append(i_bin)
    return tempList

def getPredictList(binList):
    tempList = []
    modelOut = 0
    for i in binList:
        transData = _h("".join(i))
        modelOut = model.predict(transData)
        tempList.append(modelOut)
    return tempList

def getPixList(binList):
    tempList = []
    for i in binList:
        tempList.append(_Pi_x("".join(i).count('1')))
    return tempList

def minimizeFunc():
    time_start = time.time()
    # 模擬 SHAP 問題（特徵數 p=3）
    X = np.array(samplingList_bin)  # shape: (1 samples, {featureNum} features)
    X_int = []
    for i in X:
        X_int.append(list(map(int, i)))
    X = np.array(X_int)

    y = np.array(getPredictList(samplingList_bin))     # 模型在子集 S 的值減 baseline
    w = np.array(getPixList(samplingList_bin))     # Kernel SHAP 對應權重
    
    # 2. 準備矩陣形式的數據
    # 構建特徵矩陣 X
    # 這裡我們需要將 x_data 轉換為 (n, 1) 的形狀，然後添加一列 1
    X_matrix = np.hstack((np.ones((SAMPLING_NUM, 1)), X)) # shape (n, 2)
    # 構建權重對角矩陣 W
    W_matrix = np.diag(w) # shape (n, n)
    
    initial_guess = np.arange(featureNum) # 初始猜測值 (phi_0, phi_1, ...)
    constraints = ({'type': 'eq', 'fun': equality_constraint}) # 定義約束
    # 5. 使用 scipy.optimize.minimize 進行優化
    time_start = time.time()
    result_matrix = minimize(
        objective_matrix,
        initial_guess,
        args=(X_matrix, y, W_matrix), # 傳遞給目標函數的額外矩陣參數
        method='SLSQP',
        constraints=constraints
    )
    time_end = time.time() # SHAP值計算結束時間
    print(f"計算時間={time_end - time_start}")
    return result_matrix

def getLoss(optimal_variables): # 取得跟精準SHAP值的差距
    loss = 0
    for i in range(featureNum):
        loss += (ANS_LIST[i] - optimal_variables[i])**2
    loss = math.sqrt(loss)
    print("loss in getLoss:",loss)
    return loss

def my_round(number, ndigits=0):
    p = 10**ndigits
    return (number * p * 2 + 1) // 2 / p

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

def paired(tempList):
    for i in tempList:
        if i == 2**featureNum-1: continue
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

def randomSampling(samplingNum): #  mode0: 隨機
    tempList = []
    while True:
        r = random.randint(1, 2**featureNum-2)
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
    tempList = paired(tempList)
    return tempList

def sobolSampling(samplingNum): # mode2: 低差異序列: Sobol
    x = 2**featureNum - 2
    sobolSampNum = 2
    while True: #計算 2^n >= samplingNum 的最小值
        if sobolSampNum >= samplingNum: break
        sobolSampNum *= 2
    while True:
        sobol = qmc.Sobol(d=1)  # 1-dimensional
        samples = sobol.random(n=sobolSampNum).ravel()  # Generate samples
        if sobolSampNum != samplingNum: samples = samples[:samplingNum]
        int_samples = (samples * x + 1).astype(int)
        for i in range(len(int_samples)):
            for j in range(i+1, len(int_samples)):
                if int_samples[i] == int_samples[j]:
                    int_samples[i] += np.random.choice([-1, 1])
        if len(set(int_samples)) != samplingNum : continue
        else: break
    int_samples = int_samples.tolist()
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
    int_samples = int_samples.tolist()
    return int_samples

def pairedSampling(): # mode4: 凸型配對(左右對稱)
    midNum = 2**featureNum//2
    temp = 0
    tempList = [midNum]
    # 抽樣
    i = 2
    while True:
        temp = midNum + fibonacci(i)
        if temp >= 2**featureNum-1: break
        tempList.append(temp)
        i+=1
    # 反向配對
    tempList = paired(tempList)
    return tempList

def ldFibSampling(samplingNum): # mode5: 費氏數列 + 低差異序列想法(挑選最大區間抽樣) + 配對抽樣
    top = 2**featureNum//2 - 1
    but = 2**featureNum-2 + 1
    tempList = [top, but]
    n_top = top
    n_but = but
    for _ in range(samplingNum//2):
        # 計算最大可用費氏數
        rang = n_but - n_top - 1
        maxFib = 0
        while True:
            if rang < fibonacci(maxFib):
                maxFib -= 1
                break
            else: maxFib += 1
        # 抽樣
        while True:
            ranFib = fibonacci(random.randint(math.floor(maxFib*0.4), maxFib))
            if n_top + ranFib not in tempList:
                tempList.append(n_top + ranFib)
                break
        # 找尋最大間距
        tempList.sort()
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
    tempList = paired(tempList)
    tempList.sort()
    return tempList

def pairedFibPlus(samilingNum): #mode6: 加強凸型
    top = 2**featureNum//2
    but = 2**featureNum-2
    tempList = []
    temp = 0
    i = 0
    while True:
        temp = fibonacci(i)
        temp += top
        if temp > but:
            top += 1
            i = 0
        elif temp not in tempList: tempList.append(temp)
        if len(tempList) >= samilingNum//2: break
        i += 1
    tempList = paired(tempList)
    tempList.sort()
    return tempList

def randPairedFib(samplingNum): #mode7: 隨機費氏配對
    top = 2**featureNum//2
    but = 2**featureNum-2
    tempList = []
    temp = 0
    while True:
        rand = random.randint(top, but)
        i=0
        while True:
            if len(tempList) >= samplingNum//2: break
            fbiNum = fibonacci(i)
            temp = rand + fbiNum
            if temp > but: break
            if temp not in tempList: tempList.append(temp)
            i+=1
        if len(tempList) >= samplingNum//2: break
    tempList = paired(tempList)
    tempList.sort()
    return tempList

def sampling(sampling_num, mode=0): # 選擇抽樣方法
    time_start = time.time() # 開始計算時間
    if sampling_num == "COMP_MODE":
        mode = COMP_MODE
        if mode==6: sampling_num=SAMPLING_NUM
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
    elif mode == 6:
        samplingList = pairedFibPlus(sampling_num)
    elif mode == 7:
        samplingList = randPairedFib(sampling_num)
    logging.info("結束抽樣")

    time_end = time.time() # 抽樣結束時間
    samplingList.sort()
    if len(samplingList) > sampling_num: 
        logging.warning("len(samplingList) > sampling_num")
        logging.warning(f"samplingList={samplingList}")
    return samplingList
    
def getSpac(spList): # 取得抽樣結果中各元素的距離
    spacList = []
    for i in range(1,len(spList)):
        spacList.append(abs(spList[i] - spList[i-1]))
    return spacList

def saveLossSampList(sampList):
    with open(f"{lossPath}\\ALLsamplingLossList.txt", 'a') as f:
        np.savetxt(f,sampList)

def getLOSS(): # 計算LOSS_LIMIT(COMP_MODE)
    result = minimizeFunc()
    if result.success:
        optimal_variables = result.x
        loss = getLoss(optimal_variables) # 取得和精準SHAP值之間的差距
        LOSS_LIMIT[EXPLAIN_DATA] = loss
        np.save(f"{lossPath}\\loss_mode{COMP_MODE}.npy", LOSS_LIMIT)
        np.savetxt(f"{lossPath}\\LossShapValue_{EXPLAIN_DATA}.txt", optimal_variables)
        return np.load(f"{lossPath}\\loss_mode{COMP_MODE}.npy", allow_pickle=True).item()
    else:
        print(f"優化失敗: {result.message}")

def mainFunc():
    global LOSS_LIMIT
    global samplingList
    global allLoop_loss_l2
    global countAll
    samplingList = []
    time_total = 0
    sampling_time_total = 0
    loss_total = 0
    loss_max = 0
    loss_min = 9999
    
    allLossList = []
    allSampList = []
    allSpacList = [] # 抽選子集組的各子集距離
    allShapValue = [] # 記錄每次計算的SHAP值
    
    for j in range(ROUND):
        # samplingList: 特徵子集抽樣 array = 1~2**featureNum-2
        samplingList = sampling(SAMPLING_NUM, MODE)
        samplingList_bin = toBinList(samplingList)
        
        X = np.array(samplingList_bin)  # shape: (1 samples, {featureNum} features)
        X_int = []
        for i in X:
            X_int.append(list(map(int, i)))
        X = np.array(X_int)
        y = np.array(getPredictList(samplingList_bin))     # 模型在子集 S 的值減 baseline
        w = np.array(getPixList(samplingList_bin))     # Kernel SHAP 對應權重
        # 構建特徵矩陣 X
        # 這裡我們需要將 x_data 轉換為 (n, 1) 的形狀，然後添加一列 1
        X_matrix = np.hstack((np.ones((SAMPLING_NUM, 1)), X)) # shape (n, 2)
        # 構建權重對角矩陣 W
        W_matrix = np.diag(w) # shape (n, n)
        
        initial_guess = np.arange(featureNum) # 初始猜測值 (phi_0, phi_1, ...)
        constraints = ({'type': 'eq', 'fun': equality_constraint}) # 定義約束
        # 5. 使用 scipy.optimize.minimize 進行優化
        time_start = time.time()
        result_matrix = minimize(
            objective_matrix,
            initial_guess,
            args=(X_matrix, y, W_matrix), # 傳遞給目標函數的額外矩陣參數
            method='SLSQP',
            constraints=constraints
        )
        
        time_end = time.time() # SHAP值計算結束時間
        if result_matrix.success:
            minimum_value = result_matrix.fun
            optimal_variables = result_matrix.x
            
            featureStr = f"對應的變數值: x0 = "
            resultTemp = []
            for i in range(0, featureNum): 
                featureStr += f"x{i} = {my_round(optimal_variables[i],5)}, "
                resultTemp.append(optimal_variables[i])
            featureStr = featureStr[:-2]
            
            loss = getLoss(optimal_variables) # 取得和精準SHAP值之間的差距
            loss_total += loss
            if LOSS_LIMIT.get(EXPLAIN_DATA, -1) == -1:
                LOSS_LIMIT = getLOSS()
            if loss > loss_max: loss_max = loss
            if loss < loss_min: loss_min = loss
            
            time_all_cost = time_end - time_start # 計算總耗時(抽樣時間+計算時間)
            time_total += time_all_cost
            
            allLossList.append(loss)
            allSampList.append(samplingList)# 保存全部子集組合的抽樣結果
            allSpacList.append(getSpac(samplingList))# 保存全部子集組合的子集間距離
            allShapValue.append(resultTemp) # 保存所以的ShapValue
            
            print(f"DS_NAME:{DS_NAME[DATASET]}, EXPLAIN_DATA_{EXPLAIN_DATA}, ROUND_{j}/{ROUND}||{simTime}, MODE{MODE}, SAMP{SAMPLING_NUM}")
            print(f"最小加權平方誤差: {minimum_value}")
            print(featureStr)
            print(f"中間預測值: {midPredict}")
            print(f"差距: {loss}")
            print(f"time all cost(s): {time_all_cost}s")
            print("LOSS_LIMIT=", LOSS_LIMIT)
            print(f"samplingList={samplingList}")
            print(f"len: {len(samplingList)}")
        else:
            print(f"優化失敗: {result_matrix.message}")
        print("- - - "*5)
    if ROUND != 1:
        loss_total_avg = loss_total/ROUND
        if loss_total_avg < LOSS_LIMIT[EXPLAIN_DATA]:
            countAll += 1
        if LOOPNUM > 1:
            allLoop_loss_l2 += loss_total_avg**2
        
        print(f"此為DS_NAME:{DS_NAME[DATASET]}資料集, 解釋第{EXPLAIN_DATA}筆資料, mode{MODE}, 抽樣{SAMPLING_NUM}個, 總做了{ROUND}次")
        print(f"平均抽樣時間(s): {sampling_time_total/ROUND}s")
        print(f"總時間(s): {time_total}s")
        print(f"平均差距: {loss_total_avg}")
        print(f"最大差距: {loss_max}")
        print(f"最小差距: {loss_min}")
        print("+ + + "*5)
        
        AllLossList_LOC = f"{LOCATION}\\AllLossList"
        AllList_LOC = f"{LOCATION}\\AllList"
        SpaceList_LOC = f"{LOCATION}\\SpaceList"
        AllShapValueList_LOC = f"{LOCATION}\\AllShapValueList"
        if not os.path.exists(AllLossList_LOC): os.makedirs(AllLossList_LOC)
        if not os.path.exists(AllList_LOC): os.makedirs(AllList_LOC)
        if not os.path.exists(SpaceList_LOC): os.makedirs(SpaceList_LOC)
        if not os.path.exists(AllShapValueList_LOC): os.makedirs(AllShapValueList_LOC)
        np.savetxt(f"{AllLossList_LOC}\\AllLossList_mode{MODE}_exd{EXPLAIN_DATA}_round{ROUND}_samp{SAMPLING_NUM}.txt", allLossList)
        np.savetxt(f"{AllList_LOC}\\AllList_mode{MODE}_exd{EXPLAIN_DATA}_round{ROUND}_samp{SAMPLING_NUM}.txt", allSampList)
        np.savetxt(f"{SpaceList_LOC}\\SpaceList_mode{MODE}_exd{EXPLAIN_DATA}_round{ROUND}_samp{SAMPLING_NUM}.txt", allSpacList)
        np.savetxt(f"{AllShapValueList_LOC}\\AllShapValueList_mode{MODE}_exd{EXPLAIN_DATA}_round{ROUND}_samp{SAMPLING_NUM}.txt", allShapValue)

if __name__=='__main__':
    SIMTIMES = 10
    for simTime in range(SIMTIMES):
        DATETIME_START = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=8)))
        LOOPNUM = 50 # 解釋資料數量
        DATASET = 0 # 選擇資料集
        DS_NAME = ['adult', 'airline', 'breast', 'diabetes', 'heart', 'iris']
        EXPLAIN_DATA = 0 # 選擇要解釋第幾筆資料(單筆解釋)
        MODE = 7 # 隨機方法:0, 隨機配對抽樣:1, Sobol:2, Halton:3, 凸型費氏:4, 低差異費氏配對:5, 凸型費氏+:6, 隨機費氏:7
        COMP_MODE = 6
        # 隨機選取特徵子集的數量(mode4)
        SAMPLING_NUM_LIST = [32, 34, 36, 22, 22, 14, 46]
        SAMPLING_NUM = SAMPLING_NUM_LIST[DATASET]
        ROUND = 50 # 要計算幾次
        GOLDEN_RATIO = (5**0.5 - 1)/2
        DATASET_LOC = f"SHAPSampling\\Datasets\\{DS_NAME[DATASET]}\\"
        LOCATION = f"SHAPSampling\\result_data\\{DS_NAME[DATASET]}\\simTime{simTime}\\mode{MODE}"
        ANS_LOSS_LOC = f"SHAPSampling\\result_data\\{DS_NAME[DATASET]}"
        if not os.path.exists(LOCATION): os.makedirs(LOCATION)
        LOSS_LIMIT = dict()

        totalTime_s = time.time()
        reCalcu = False #是否重新計算ANS_LIST
        ansPath = f"{ANS_LOSS_LOC}\\ANS"
        lossPath = f"{ANS_LOSS_LOC}\\LOSS"
        fibonacciSeq = {0:0, 1:1}

        countAll = 0
        allLoop_loss_l2 = 0

        X_train, X_test, y_train, y_test = _setData()
        # Number of features(M)
        columns = X_train.columns.tolist()
        featureNum = len(columns)
        SAMPLING_NUM = 4*featureNum

        model = Model()

        print(f"StartTime={DATETIME_START}")
        #if SAMPLING_NUM >= 2**featureNum: SAMPLING_NUM = 2**featureNum-2
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
                samplingList_bin = toBinList(samplingList)
                LOSS_LIMIT = getLOSS()
            else:
                with open(lossPath + f"\\loss_mode{COMP_MODE}.npy", 'rb') as file:
                    LOSS_LIMIT = np.load(file, allow_pickle=True).item() # 字典[EXPLAIN_DATA] 保存上限設定值(mode4)
                if LOSS_LIMIT.get(EXPLAIN_DATA, -1) <= 0:
                    samplingList = sampling("COMP_MODE")
                    samplingList_bin = toBinList(samplingList)
                    LOSS_LIMIT = getLOSS()

            print("ANS_LIST=",ANS_LIST)
            print("LOSS_LIMIT=",LOSS_LIMIT) 

            if MODE == COMP_MODE: ROUND = 1
            mainFunc()
            EXPLAIN_DATA += 1
            
        totalTime_e = time.time()
        DATETIME_END = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=8)))
        print("* * * "*5)
        print(f"LOOPNUM_{LOOPNUM}, ROUND_{ROUND}, DS_NAME:{DS_NAME[DATASET]}, MODE{MODE}")
        print(f"StartTime={DATETIME_START}")
        print(f"EndTime={DATETIME_END}")
        print(f"總花費時間: {(totalTime_e-totalTime_s)/60}m")
        print("countAll =",countAll)
        print("allLoop_loss_l2 =", math.sqrt(allLoop_loss_l2))
        print("* * * "*5)