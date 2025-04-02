# import csv
# import re
# import os

# def extract_model_name(log_filename):
#     """ 从日志文件名中提取模型名称（去掉扩展名 .log） """
#     return os.path.splitext(log_filename)[0]  # 去除 .log 后缀

# def parse_log_file(log_file):
#     """ 解析单个日志文件，提取各项指标 """
#     data = []
#     model_name = extract_model_name(os.path.basename(log_file))  # 提取模型名称
#     with open(log_file, "r", encoding="utf-8") as file:
#         content = file.read()

#     # 正则表达式匹配各项指标
#     metrics = {
#         "Matching Score": re.search(r"========== Matching Score Summary =========.*?Mean: ([\d.]+) CInterval: ([\d.]+)", content, re.S),
#         "FID": re.search(r"========== FID Summary =========.*?Mean: ([\d.]+) CInterval: ([\d.]+)", content, re.S),
#         "Diversity": re.search(r"========== Diversity Summary =========.*?Mean: ([\d.]+) CInterval: ([\d.]+)", content, re.S),
#         "MultiModality": re.search(r"========== MultiModality Summary =========.*?Mean: ([\d.]+) CInterval: ([\d.]+)", content, re.S)
#     }

#     # 正则表达式匹配 R_precision Summary 中的 top k 数据
#     r_precision_matches = re.findall(r"\(top (\d+)\).*?Mean: ([\d.]+) CInt: ([\d.]+)", content)

#     # 处理匹配到的指标数据
#     for metric, match in metrics.items():
#         if match:
#             data.append([model_name, metric, float(match.group(1)), float(match.group(2))])

#     # 处理 R_precision（top 1, top 2, top 3）
#     for match in r_precision_matches:
#         top_k, mean, cint = match
#         data.append([model_name, f"R_precision (top {top_k})", float(mean), float(cint)])

#     return data

# def process_logs_in_directory(log_dir, output_csv):
#     """ 遍历目录下的所有日志文件并写入 CSV """
#     all_data = [["Action_Types", "Metric", "Mean", "Confidence Interval"]]  # CSV 头部

#     for filename in os.listdir(log_dir):
#         if filename.endswith(".log"):  # 只处理 .log 文件
#             log_path = os.path.join(log_dir, filename)
#             all_data.extend(parse_log_file(log_path))

#     # 写入 CSV
#     with open(output_csv, mode="w", newline="") as file:
#         writer = csv.writer(file)
#         writer.writerows(all_data)

#     print(f"✅ CSV 文件已生成: {output_csv}")

# # 设置日志文件所在目录
# log_directory = "//liujinxin/code/text-to-motion/log/hu_GPRO_union15_3-11-21-12-57/checkpoint-36000"  # 🔹 请修改为你的日志目录路径
# output_csv_path = "//liujinxin/code/text-to-motion/log/hu_GPRO_union15_3-11-21-12-57/checkpoint-36000/hu_GPRO_union15_3-11-21-12-57.csv"

# # 处理日志文件并生成 CSV
# process_logs_in_directory(log_directory, output_csv_path)
import csv
import re
import os

def extract_model_name(log_filename):
    """ 从日志文件名中提取模型名称（去掉扩展名 .log） """
    return os.path.splitext(log_filename)[0]  # 去除 .log 后缀

def parse_log_file(log_file):
    """ 解析单个日志文件，提取各项指标 """
    model_name = extract_model_name(os.path.basename(log_file))  # 提取模型名称
    with open(log_file, "r", encoding="utf-8") as file:
        content = file.read()

    # 正则表达式匹配各项指标
    metrics = {
        "Matching Score": re.search(r"========== Matching Score Summary =========.*?Mean: ([\d.]+) CInterval: ([\d.]+)", content, re.S),
        "FID": re.search(r"========== FID Summary =========.*?Mean: ([\d.]+) CInterval: ([\d.]+)", content, re.S),
        "Diversity": re.search(r"========== Diversity Summary =========.*?Mean: ([\d.]+) CInterval: ([\d.]+)", content, re.S),
        "MultiModality": re.search(r"========== MultiModality Summary =========.*?Mean: ([\d.]+) CInterval: ([\d.]+)", content, re.S)
    }

    # 正则表达式匹配 R_precision Summary 中的 top k 数据
    r_precision_matches = re.findall(r"\(top (\d+)\).*?Mean: ([\d.]+) CInt: ([\d.]+)", content)

    # 初始化数据字典
    data = {"Action_Types": model_name}

    # 处理匹配到的指标数据
    for metric, match in metrics.items():
        if match:
            data[f"{metric} Mean"] = float(match.group(1))
            data[f"{metric} Confidence Interval"] = float(match.group(2))

    # 处理 R_precision（top 1, top 2, top 3）
    for match in r_precision_matches:
        top_k, mean, cint = match
        data[f"R_precision (top {top_k}) Mean"] = float(mean)
        data[f"R_precision (top {top_k}) Confidence Interval"] = float(cint)

    return data

def process_logs_in_directory(log_dir, output_csv):
    """ 遍历目录下的所有日志文件并写入 CSV """
    all_data = []

    # 获取所有日志文件的指标数据
    for filename in os.listdir(log_dir):
        if filename.endswith(".log"):  # 只处理 .log 文件
            log_path = os.path.join(log_dir, filename)
            all_data.append(parse_log_file(log_path))

    # 获取所有可能的列名
    columns = set()
    for row in all_data:
        columns.update(row.keys())
    columns = sorted(columns)  # 按字母顺序排序

    # 写入 CSV
    with open(output_csv, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_data)

    print(f"✅ CSV 文件已生成: {output_csv}")

# 设置日志文件所在目录
# log_directory = "/liujinxin/code/text-to-motion/log/hu_GPRO_union15_3-17-19-45-28/checkpoint-1000"  # 🔹 请修改为你的日志目录路径
# output_csv_path = "/liujinxin/code/text-to-motion/log/hu_GPRO_union15_3-17-19-45-28/checkpoint-1000/28.csv"
log_directory = "/liujinxin/code/text-to-motion/log/hu_GPRO_union15_3-19-3-5-27/checkpoint-1200"
output_csv_path = '/liujinxin/code/text-to-motion/log/hu_GPRO_union15_3-19-3-5-27/base.csv'
# 处理日志文件并生成 CSV
process_logs_in_directory(log_directory, output_csv_path)