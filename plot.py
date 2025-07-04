import random
import json
import os
import math

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

DS_ID = 0
DS_NAME = ['adult', 'airline', 'breast', 'diabetes', 'heart', 'iris', 'IEAClassification']
DS_SAMPNUM = [56, 88, 120, 68, 44, 0, 116]
MODE4_SAMP = [40, 62, 86, 48, 32, 0, 84]
SIM_TIME = 0
MODE = 6 # 隨機方法:0, 隨機配對抽樣:1, Sobol:2, Halton:3, 凸型費氏:4, 低差異費氏配對:5, 凸型費氏+:6, 隨機費氏:7, 倍數費氏:8
COMP_MODE = 6
ROUND = 50
SAMP = DS_SAMPNUM[DS_ID]
if MODE == 4:SAMP=MODE4_SAMP[DS_ID]
LOCATION = f"SHAPSampling\\result_data\\{DS_NAME[DS_ID]}"

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
            print("dataset:", DS_NAME[DS_ID], "MODE:", MODE, "simTime:", simTime, "avgL2 =", my_round(avgL2, 3))
            i = 0
            avgL2 = 0
            loss = 0
            simTime+=1
        
        i+=1
        if not os.path.exists(LOCATION + f"\\simTime{simTime}"): break

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

# get_AllLossList_L2()
SHAPvalL2_sampListGapL2_scatter()