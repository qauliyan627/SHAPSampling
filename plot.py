import random
import json

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

LOCATION = "SHAPSampling\\plot_data\\"

class GetData:
    def mergeData(self, twList): # 轉為一維陣列
        temp = twList.flatten()
        return temp
    
    def txtListData(self, dataName): # 取得List資料
        temp = []
        listData = np.loadtxt(fname=LOCATION+dataName)
        for i in listData:
            if i.tolist() not in temp:
                temp.append(i.tolist())
        listData = temp.copy()
        return listData
    
    def jsonDictData(self, dataName):
        with open(LOCATION+dataName) as json_file:
            dictData = json.load(json_file)
            return dictData

class ShapResultPlot:
    def subsetListBox(self, allsubsetList): # 繪製盒狀圖(多個特徵值組合)
        subsetLabels = [f"ss{i+1}" for i in range(len(allsubsetList))] # 子集的標籤，可以根據您的需求自訂
        plt.boxplot(allsubsetList, label=subsetLabels)
        plt.tight_layout()
        plt.show()
    
    def spacListBar(self, allSpacList): # 繪製棒狀圖(多個特徵值組合)
        randomNum = random.randint(0,len(allSpacList))
        indexRange = range(1, len(allSpacList[randomNum])+1)
        plt.tick_params(axis='x', labelsize=20)
        plt.xticks(indexRange)
        plt.bar(indexRange,allSpacList[randomNum])
        plt.show()
        
    def countListBar(self, smpList):
        temp = []
        randomDict = {}
        for i in smpList:
            for j in i:
                randomDict[j] = randomDict.get(j, 0)+1
                temp.append(j)
        
        plotB = SET_SIZE//23
        plotB = SET_SIZE//32
        tempDict = {}
        for i in range(1, SET_SIZE, plotB):
            for j in range(i, i+plotB):
                if i+plotB >= SET_SIZE:
                    tempDict[f'{i}-{SET_SIZE-1}'] = tempDict.get(f'{i}-{SET_SIZE-1}', 0)+randomDict.get(j, 0)
                else:
                    tempDict[f'{i}-{i+plotB-1}'] = tempDict.get(f'{i}-{i+plotB-1}', 0)+randomDict.get(j, 0)
        
        plt.tick_params(axis='x', labelsize=10)
        plt.xticks(rotation=-20)
        plt.bar(tempDict.keys(),tempDict.values())
        plt.show()
    
    def countFeatureBar(self, smpList): # 計算各特徵被抽到的次數
        FeatureDict = {}
        for i in smpList:
            for j in i:
                count = 1
                j_bin = format(int(j), 'b')
                j_bin = j_bin.zfill(11)
                for k in j_bin:
                    if k == '1':
                        FeatureDict[f"F{count}"] = FeatureDict.get(f"F{count}", 0)+1
                    count+=1
        a = {}
        for i in range(1,12):
            a[f'F{i}'] = FeatureDict[f"F{i}"]
        FeatureDict = a
        print(len(smpList)*len(smpList[0]))
        print(FeatureDict)
        plt.tick_params(axis='x', labelsize=10)
        plt.xticks(rotation=-20)
        plt.bar(FeatureDict.keys(),FeatureDict.values())
        plt.show()



def drowGapShapValue():
    gapDataNAME = "AllGapList"
    shapDataNAME = "AllShapValueList"
    getdata = GetData()

    gapData = getdata.txtListData(f"{gapDataNAME}_mode{MODE}_round{ROUND}.txt")
    shapValueData = getdata.txtListData(f"{shapDataNAME}_mode{MODE}_round{ROUND}.txt")
    
    for i in range(len(shapValueData[0])):
        temp = []
        for j in shapValueData:
            temp.append(j[i])
        plt.scatter(gapData, temp)
    plt.show()
    
def drowFeatureDistribution(listData): # 繪製出特徵分布狀態 AllList
    for _ in range(100):
        randNum = random.randint(0,len(listData)-1)
        plt.bar(listData[randNum], range(len(listData[randNum])))
    plt.show()

SET_SIZE = 2**11

getPlot = ShapResultPlot()
getdata = GetData()

# 隨機方法0, 傳統費氏(凹型)1, 黃金抽樣2, 平均費氏3, 對稱費氏(凸型)4, 分層費氏5
LISTNAME = "GapSampList" # AllList GapSampList SpaceList AllGapList
MODE = 2
ROUND = 100

fileName = f"{LISTNAME}_mode{MODE}_round{ROUND}.txt"
fileName = "AllSamplingGapList.txt"

data = getdata.txtListData(fileName)
#drowFeatureDistribution(data)
getPlot.countListBar(data)
#getPlot.countFeatureBar(data)
#drowGapShapValue()