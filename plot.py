import random
import json
import os
import math

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

ID = 186
MODE = 5
ROUND = 50
LOCATION = f"SHAPSampling\\result_data\\{ID}"

def getAll_AllLossList_L2():# 計算AllLossList的L2
    i = 0
    avgL2 = 0
    loss = 0
    while True:
        if os.path.exists(LOCATION + f"\\mode{MODE}\\AllLossList" + f"\\AllLossList_mode{MODE}_exd{i}_round{ROUND}.txt"):
            #輸入AllLossList
            allLossList = np.loadtxt(LOCATION + f"\\mode{MODE}\\AllLossList" + f"\\AllLossList_mode{MODE}_exd{i}_round{ROUND}.txt")
            for j in allLossList:
                loss += j**2
            loss = math.sqrt(loss)
            avgL2 += loss
            i+=1
        else: 
            print("avgL2 =", avgL2)
            break
def get_AllLossList_L2(i):# 計算單筆AllLossList的L2
    loss = 0
    if os.path.exists(LOCATION + f"\\mode{MODE}\\AllLossList" + f"\\AllLossList_mode{MODE}_exd{i}_round{ROUND}.txt"):
        #輸入AllLossList
        allLossList = np.loadtxt(LOCATION + f"\\mode{MODE}\\AllLossList" + f"\\AllLossList_mode{MODE}_exd{i}_round{ROUND}.txt")
        for j in allLossList:
            loss += j**2
        loss = math.sqrt(loss)
    return loss


loss = get_AllLossList_L2(0)
getAll_AllLossList_L2()