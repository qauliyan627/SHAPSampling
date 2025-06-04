import numpy as np

DATASET = 1 # 選擇資料集
COMP_MODE = 4
ID = [186, 519, 563, 1, 165, 60, 544]
LOCATION = f"SHAPSampling\\plot_data\\{ID[DATASET]}"
gapPath = f"{LOCATION}\\GAP"
with open(gapPath + f"\\gap_mode{COMP_MODE}.npy", 'rb') as file:
    GAP_LIMIT = np.load(file, allow_pickle=True).item()

mode4all = 0
count = 0
for i in GAP_LIMIT.values():
    count += 1
    mode4all += i
print(GAP_LIMIT)
print(mode4all/count)
