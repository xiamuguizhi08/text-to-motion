import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# 示例数据1和数据2
data1 = np.random.random((T1, 3))  # 假设数据1的形状是 (T1, 3)
data2 = np.random.random((T2, 3))  # 假设数据2的形状是 (T2, 3)

# 计算fastdtw距离
distance, path = fastdtw(data1, data2, dist=euclidean)

print(f"FastDTW distance: {distance}")
