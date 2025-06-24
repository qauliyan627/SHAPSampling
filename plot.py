import random
import json
import os
import math

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

DATASET_NAME = "breast"
SIM_TIME = 0
MODE = 6
COMP_MODE = 6
ROUND = 50
SAMP = 120
LOCATION = f"SHAPSampling\\result_data\\{DATASET_NAME}"
if MODE == 6: ROUND = 1
if MODE == 4: SAMP = 48

def my_round(number, ndigits=0):
    p = 10**ndigits
    return (number * p * 2 + 1) // 2 / p

def getAll_AllLossList_L2():# 計算AllLossList的L2
    simTime = SIM_TIME
    i = 0
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
            i+=1
        else:
            avgL2 = math.sqrt(avgL2)
            print("dataset:", DATASET_NAME, "MODE:", MODE, "simTime:", simTime, "avgL2 =", my_round(avgL2, 3))
            i = 0
            avgL2 = 0
            loss = 0
            simTime+=1
        if not os.path.exists(LOCATION + f"\\simTime{simTime}"): break

def get_AllLossList_L2(i):# 計算單筆AllLossList的L2
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
    print("DATASET_NAME:", DATASET_NAME, "MODE:", MODE)
    print(sum(lossLimit.values())/len(lossLimit))
    print(loss)

getAll_AllLossList_L2()