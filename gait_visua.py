# 生成多个98个1，98个2，98个3的序列
import pandas as pd

n = 58 # 序列的数量
result = []

for i in range(1, n + 1):
    result.extend([i] * 983)

result=pd.DataFrame(result)
result.to_csv('n_r_number.csv')
print(result)
