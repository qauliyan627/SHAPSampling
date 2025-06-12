import numpy as np

DATASET = 2 # 選擇資料集
COMP_MODE = 4
ID = [186, 519, 563, 1, 165, 60, 544]
LOCATION = f"SHAPSampling\\result_data\\{ID[DATASET]}"
gapPath = f"{LOCATION}\\LOSS"
with open(gapPath + f"\\loss_mode{COMP_MODE}.npy", 'rb') as file:
    GAP_LIMIT = np.load(file, allow_pickle=True).item()

mode4all = 0
count = 0
for i in GAP_LIMIT.values():
    count += 1
    mode4all += i
print(GAP_LIMIT)
print(mode4all/count)
