import math

def golden_ratio_lds_i(index, n):
  """
  使用索引產生 [1, n] 區間的黃金比例 LDS 整數樣本。

  Args:
    index: 樣本的索引 (從 0 開始)。
    n: 目標整數區間的上界。

  Returns:
    一個 [1, n] 區間的整數樣本。
  """
  golden_ratio = 1.61803398875
  lds_value = math.fmod(golden_ratio * float(index), 1.0)
  return math.floor(lds_value * n) + 1

def golden_ratio_lds_next_i(last_float, n):
  """
  使用前一個值產生下一個 [1, n] 區間的黃金比例 LDS 整數樣本。

  Args:
    last_float: 上一個產生的整數樣本。
    n: 目標整數區間的上界。

  Returns:
    下一個 [1, n] 區間的整數樣本。
  """
  golden_ratio_fraction = 0.61803398875
  next_float = math.fmod((last_float - 1) / n + golden_ratio_fraction, 1.0)
  return math.floor(next_float * n) + 1

if __name__ == "__main__":
  n = 100  # 設定目標區間為 [1, 10]

  print(f"使用索引產生 [1, {n}] 的黃金比例 LDS 樣本：")
  for i in range(10):
    print(golden_ratio_lds_i(i, n), end=" ")
  print()

  print(f"使用前一個值產生 [1, {n}] 的黃金比例 LDS 樣本：")
  last_value = 0
  for _ in range(10):
    last_value = golden_ratio_lds_next_i(last_value, n)
    print(last_value, end=" ")
  print()