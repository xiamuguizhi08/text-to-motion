import os
import numpy as np

# 获取当前路径
current_path = '/liujinxin/code/text-to-motion/dataset/amass_15_1000_2/hu_GPRO_union15_3-17-19-45-28/checkpoint-1000/test_caption/folder_2'


# 记录最大 T 及对应的文件路径
max_T = -1
max_file = None
shape_list = []

# 遍历当前目录下的所有 .npy 文件
for file in os.listdir(current_path):
    if file.endswith(".npy"):
        file_path = os.path.join(current_path, file)
        
        # 加载 .npy 文件
        try:
            data = np.load(file_path)
            # 检查维度是否符合 (T, 15, 3)
            if data.shape[1:] == (15, 3):
                T = data.shape[0]
                shape_list.append(T)
                print(f'文件: {file_path}, shape: ({T}, 15, 3)')
                
                # 记录最大 T 的文件
                if T > max_T:
                    max_T = T
                    max_file = file_path
        except Exception as e:
            print(f"加载文件 {file_path} 失败: {e}")

# 计算 shape_list 的均值
if shape_list:
    mean_T = sum(shape_list) / len(shape_list)
    print(f'所有符合条件的 T 值: {shape_list}')
    print(f'均值: {mean_T:.2f}')
    print(f'最大 T 的文件: {max_file}, 维度: ({max_T}, 15, 3)')
else:
    print("未找到符合维度 (T, 15, 3) 的 .npy 文件")
