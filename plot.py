import random
import json
import os
import math
import copy

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
plt.rcParams['font.family'] = 'DFKai-SB'
import seaborn as sns
import numpy as np

def my_round(number, ndigits=0):
    p = 10**ndigits
    return (number * p * 2 + 1) // 2 / p

def getL2(numList):
    l2 = 0
    for i in numList:
        l2 += i**2
    l2 = math.sqrt(l2)
    return l2
        
def get_AllLossList_L2():# 計算AllLossList的L2
    simTime = SIM_TIME
    i = 1
    avgL2_List = []
    avgL2 = 0
    loss = 0
    while True:
        allLossList_Path = LOCATION + f"\\simTime{simTime}\\mode{MODE}\\AllLossList\\AllLossList_mode{MODE}_exd{i}_round{ROUND}_samp{SAMP}.txt"
        if os.path.exists(allLossList_Path):
            #輸入AllLossList
            allLossList = np.loadtxt(allLossList_Path, ndmin=1)
            for j in allLossList:
                loss += j
            loss = loss/len(allLossList)
            
            avgL2 += loss**2
            loss = 0
        else:
            avgL2 = math.sqrt(avgL2)
            avgL2_List.append(avgL2)
            print("dataset:", DS_NAME[DS_ID], "MODE:", MODE, "simTime:", simTime, "avgL2 =", my_round(avgL2, 3))
            i = 0
            avgL2 = 0
            loss = 0
            simTime+=1
        
        i+=1
        if not os.path.exists(LOCATION + f"\\simTime{simTime}"): break
    return avgL2_List

def get_OneLossList_L2(i):# 計算單筆AllLossList的L2
    loss = 0
    if os.path.exists(LOCATION + f"\\mode{MODE}\\AllLossList" + f"\\AllLossList_mode{MODE}_exd{i}_round{ROUND}.txt"):
        #輸入AllLossList
        allLossList = np.loadtxt(LOCATION + f"\\mode{MODE}\\AllLossList" + f"\\AllLossList_mode{MODE}_exd{i}_round{ROUND}.txt")
        for j in allLossList:
            loss += j**2
        loss = math.sqrt(loss)
    return loss

def getLOSS(): # 計算lossLimit的L2
    loss = 0
    lossLimit = np.load(f"{LOCATION}\\LOSS\\loss_mode{COMP_MODE}.npy", allow_pickle=True).item()
    for j in lossLimit.values():
        loss += j**2
    loss = math.sqrt(loss)
    print("DS_NAME[DS_ID]:", DS_NAME[DS_ID], "MODE:", MODE)
    print(sum(lossLimit.values())/len(lossLimit))
    print(loss)

def SHAPvalL2_sampListGapL2_scatter():
    simTime = SIM_TIME
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    while True:
        print(f"simTime{simTime}")
        dataLoc = LOCATION + f"\\simTime{simTime}\\mode{MODE}\\"
        if not os.path.exists(dataLoc): break
        exd = 0
        allLossList_Big = []
        spaceListL2_Big = []
        while True:
            dataLoc_allLossList = dataLoc + "AllLossList\\" + f"AllLossList_mode{MODE}_exd{exd}_round{ROUND}_samp{SAMP}.txt"
            dataLoc_spaceList = dataLoc + "SpaceList\\" + f"SpaceList_mode{MODE}_exd{exd}_round{ROUND}_samp{SAMP}.txt"
            if not os.path.exists(dataLoc_allLossList): break
            # 取得資料
            allLossList = np.loadtxt(dataLoc_allLossList)
            spaceList = np.loadtxt(dataLoc_spaceList)
            spaceListL2 = []
            # 計算每筆資料的L2
            for i in spaceList:
                spaceL2 = getL2(i)
                spaceListL2.append(spaceL2)
            allLossList_Big.extend(allLossList)
            spaceListL2_Big.extend(spaceListL2)
            exd+=1
        
        plt.scatter(allLossList_Big,spaceListL2_Big)
        plt.title('scatter plot')
        plt.xlabel('allLossList')
        plt.ylabel('spaceListL2')
        plt.show()
        simTime+=1
        
def SHAPvalL2_AllListLayerL2_scatter():
    simTime = SIM_TIME
    while True:
        print(f"simTime{simTime}")
        dataLoc = LOCATION + f"\\simTime{simTime}\\mode{MODE}\\"
        if not os.path.exists(dataLoc): break
        exd = 0
        allLossList_Big = []
        spaceListL2_Big = []
        while True:
            dataLoc_allLossList = dataLoc + "AllLossList\\" + f"AllLossList_mode{MODE}_exd{exd}_round{ROUND}_samp{SAMP}.txt"
            dataLoc_spaceList = dataLoc + "SpaceList\\" + f"SpaceList_mode{MODE}_exd{exd}_round{ROUND}_samp{SAMP}.txt"
            if not os.path.exists(dataLoc_allLossList): break
            # 取得資料
            allLossList = np.loadtxt(dataLoc_allLossList)
            spaceList = np.loadtxt(dataLoc_spaceList)
            spaceListL2 = []
            # 計算每筆資料的L2
            for i in spaceList:
                spaceL2 = getL2(i)
                spaceListL2.append(spaceL2)
            allLossList_Big.extend(allLossList)
            spaceListL2_Big.extend(spaceListL2)
            exd+=1
        allLossList_Loss = []
        spaceListL2_Loss = []
        for i in range(len(allLossList_Big)):
            if allLossList_Big[i] < 0.2:
                allLossList_Loss.append(allLossList_Big[i])
                spaceListL2_Loss.append(spaceListL2_Big[i])
                
        plt.scatter(allLossList_Loss,spaceListL2_Loss)
        plt.title('scatter plot')
        plt.xlabel('allLossList')
        plt.ylabel('spaceListL2')
        plt.show()
        simTime+=1

def AllLossList_L2_bar():
    totalAllLossList = [] # [[mode1各資料集結果], [mode2各資料集結果], ...]
    for mode in range(9):
        allLossList_ds = []
        for id in range(len(DS_NAME)):
            if mode == 4: samp = MODE4_SAMP[id]
            else: samp = DS_SAMPNUM[id]
            Location = f"SHAPSampling\\result_data\\{DS_NAME[id]}"
            allLoss_avg = 0
            avgCount = 0
            for sim in range(10):
                for exd in range(50):
                    allLossList = np.loadtxt(Location + f"\\simTime{sim}\\mode{mode}\\AllLossList\\AllLossList_mode{mode}_exd{exd}_round{ROUND}_samp{samp}.txt")
                    allLoss_avg += sum(allLossList)/len(allLossList)
                    avgCount += 1
            allLossList_ds.append(allLoss_avg/avgCount)
        totalAllLossList.append(allLossList_ds)
    print(totalAllLossList)
    #將totalAllLossList轉換成比例關係
    totalAllLossListP = copy.deepcopy(totalAllLossList)
    for ds in range(len(DS_NAME)):
        temp = 0
        for mode in range(len(MODES)):
            temp += totalAllLossList[mode][ds]
        for mode in range(len(MODES)):
            totalAllLossListP[mode][ds] = (totalAllLossList[mode][ds]/temp)*100
    
    #繪圖
    ds_name = ['heart', 'adult', 'diabetes', 'airline', 'IEA', 'breast']
    ds_featureNum = ["11","14","17","22","29","31"]
    hatchs = ["X", "||", "oo", "//", "**", "XX", "--", "..", "++", "XXX"]
    markers = ["o", "v", "*", "s", "p", "h", "x", "d", "|"]
    linestyles = ["--", "-.", ":"]
    x = [i+1 for i in range(len(ds_name))]
    y = [i for i in range(1,4)]
    y.extend([i for i in range(4,11,2)])# [4 6 8 10]
    y.extend([i for i in range(15,21,5)])# [15 20]
    y.extend([i for i in range(20,51,10)])
    #y = [i for i in range(2,42,2)]
    width = 0.1

    fig, ax = plt.subplots()
    for i in range(len(totalAllLossList)):
        ax.bar([p + i*width for p in x], totalAllLossList[i], label=MODES[i], width=width, hatch=hatchs[i], color='w', edgecolor='k')
        #ax.plot([p + len(totalAllLossList)//2*width for p in x], totalAllLossList[i], label=MODES[i], marker=markers[i], markersize=15.0, linestyle=":", linewidth=4, color='k')
    
    ax.set_yscale("log", base=3)
    ax.set_yticks(y)
    ax.set_yticklabels([str(val) for val in y], fontsize=40)
    plt.xticks([p + len(totalAllLossList)//2*width for p in x], [ds_name[i]+f"({ds_featureNum[i]})" for i in range(len(ds_name))], fontsize=40) #設定 X 軸刻度標籤
    plt.legend(loc='upper left', ncol=len(MODES)//4, fontsize=40 ) #顯示圖例
    plt.title(f'各方法於資料集之L2-norm總結果(log_3)', fontsize=50) #設定標題
    plt.ylabel('L2-norm', fontsize=40)
    plt.grid()
    plt.show()

def SamplingGapList_scatter(): #抽樣結果的散圖 比較不同方法的抽樣分布
    ds_id = 2
    ds_name = ['heart', 'adult', 'diabetes', 'airline', 'IEA', 'breast']
    Location = f"SHAPSampling\\result_data\\{ds_name[ds_id]}"
    allSpaceList = []
    allSampList = []
    # [mode0:[[子集組合間隔1],[子集組合間隔2]], mode2:[], mode3:[]]
    for mode in range(9):
        if mode == 4: samp = MODE4_SAMP[ds_id]
        else: samp = DS_SAMPNUM[ds_id]
        spaceList_mode = []
        sampList_mode = []
        for sim in range(1):
            for exd in range(0,50,10):
                print(f"mode{mode}, sim{sim}, exd{exd}")
                spaceList = np.loadtxt(Location + f"\\simTime{sim}\\mode{mode}\\SpaceList\\SpaceList_mode{mode}_exd{exd}_round{ROUND}_samp{samp}.txt")
                spaceList_mode.append(spaceList)
                sampList = np.loadtxt(Location + f"\\simTime{sim}\\mode{mode}\\AllList\\AllList_mode{mode}_exd{exd}_round{ROUND}_samp{samp}.txt")
                sampList = [(np.array(i)-min(i)).tolist() for i in sampList]
                sampList_mode.append(sampList)
        allSpaceList.append(spaceList_mode)
        allSampList.append(sampList_mode)
    print("開始製圖")
    fig, ax = plt.subplots()
    for mode in range(9):
        print(f"繪製mode{mode}")
        for spaceList in allSpaceList[mode]:
            x_mode = []
            for i in spaceList:
                rand = random.uniform(-0.3,0.3)
                x_mode.append([mode+1+rand for _ in range(len(i))])
            ax.scatter(x_mode, spaceList, marker='.', color='k')
    plt.show()
    fig, ax = plt.subplots()
    for mode in range(9):
        print(f"繪製mode{mode}")
        for sampList in allSampList[mode]:
            x_mode = []
            for i in sampList:
                rand = random.uniform(-0.3,0.3)
                x_mode.append([mode+1+rand for _ in range(len(i))])
            ax.scatter(x_mode, sampList, marker='.', color='k')
    plt.show()

def shapValue_box_new():# SHAP值的盒狀圖
    ds_id = 5
    ds_name = ['heart', 'adult', 'diabetes', 'airline', 'IEA', 'breast']
    Location = f"SHAPSampling\\result_data\\{ds_name[ds_id]}"
    allLossList = []
    LossList_x = []
    # [mode0:[[子集組合間隔1],[子集組合間隔2]], mode2:[], mode3:[]]
    for mode in range(9):
        if mode == 4: samp = MODE4_SAMP[ds_id]
        else: samp = DS_SAMPNUM[ds_id]
        lossList_mode = []
        for sim in range(10):
            Loss_avg = 0
            for exd in range(0,50):
                print(f"mode{mode}, sim{sim}, exd{exd}")
                lossList = np.loadtxt(Location + f"\\simTime{sim}\\mode{mode}\\AllLossList\\AllLossList_mode{mode}_exd{exd}_round{ROUND}_samp{samp}.txt")
                for i in lossList: lossList_mode.append(i)
                Loss_avg = sum(lossList)/len(lossList)
            #lossList_mode.append(Loss_avg)
        allLossList.append(lossList_mode)
        if mode==4 or mode==6 or mode==7: LossList_x.append(lossList_mode)
    print(allLossList, LossList_x)
    fig, ax = plt.subplots()
    ax.boxplot(allLossList, tick_labels=MODES, whis=2)
    plt.yticks(fontsize=40)
    plt.xticks(fontsize=40) #設定 X 軸刻度標籤
    plt.title(f'{ds_name[ds_id]}', fontsize=50)
    plt.ylabel('L2-norm', fontsize=40)
    plt.show()
    
    fig, ax = plt.subplots()
    modes = ["凸型費氏", "位移凸型a", "位移凸型b"]
    ax.boxplot(LossList_x, tick_labels=modes, whis=2)
    y = [i for i in range(4,6)]
    plt.yticks(fontsize=40)
    plt.xticks(fontsize=40) #設定 X 軸刻度標籤
    plt.title(f'{ds_name[ds_id]}', fontsize=50)
    plt.ylabel('L2-norm', fontsize=40)
    plt.show()

def shapValue_box():# SHAP值的盒狀圖
    for i in range(0,9):
        fig, ax = plt.subplots()
        samp = SAMP
        if i == 4:samp=MODE4_SAMP[DS_ID]
        k=0
        j=0
        location = LOCATION + f"\\simTime{k}\\mode{i}\\AllLossList\\" + f"AllLossList_mode{i}_exd{j}_round{ROUND}_samp{samp}.txt"
        allLossList = [] #len=10
        allLossList_xList = []
        xList = []
        while os.path.exists(location):
            allLoss = []
            while os.path.exists(location):#
                tempList = np.loadtxt(location)
                for temp in tempList:
                    allLoss.append(temp)
                j+=1
                location = LOCATION + f"\\simTime{k}\\mode{i}\\AllLossList\\" + f"AllLossList_mode{i}_exd{j}_round{ROUND}_samp{samp}.txt"
            j=0
            k+=1
            location = LOCATION + f"\\simTime{k}\\mode{i}\\AllLossList\\" + f"AllLossList_mode{i}_exd{j}_round{ROUND}_samp{samp}.txt"
        allLossList_xList = [[j+1]*len(allLossList[0]) for j in range(len(allLossList))]
        
        ax.boxplot(allLossList, whis=2)
        plt.yticks(fontsize=40)
        plt.xticks(fontsize=40) #設定 X 軸刻度標籤
        plt.title(f'{MODES[i]} {DS_NAME[DS_ID]}', fontsize=50)
        plt.legend(loc='upper center', ncol=9, fontsize=40) #顯示圖例
        plt.ylabel('L2-norm', fontsize=40)
        plt.show()
    pass



DS_ID = 1
MODES = ["隨機方法", "配對隨機", "Sobol", "Halton", "凸型費氏", "低差異費氏", "位移凸型a", "位移凸型b", "倍數費氏"]
DS_NAME = ['heart', 'adult', 'diabetes', 'airline', 'IEAClassification', 'breast']
DS_SAMPNUM = [44, 56, 68, 88, 116, 120]
MODE4_SAMP = [32, 40, 48, 62, 84, 86]
SIM_TIME = 0
MODE = 0 # 隨機方法:0, 隨機配對抽樣:1, Sobol:2, Halton:3, 凸型費氏:4, 低差異費氏配對:5, 凸型費氏+:6, 隨機費氏:7, 倍數費氏:8
COMP_MODE = 6
ROUND = 50
SAMP = DS_SAMPNUM[DS_ID]
if MODE == 4:SAMP=MODE4_SAMP[DS_ID]
LOCATION = f"SHAPSampling\\result_data\\{DS_NAME[DS_ID]}"

#get_AllLossList_L2()
#AllLossList_L2_bar()
#SHAPvalL2_sampListGapL2_scatter()
#SamplingGapList_scatter()
shapValue_box_new()