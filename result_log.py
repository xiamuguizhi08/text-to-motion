import os
import csv
import re

# 日志文件所在目录
log_dir = "/liujinxin/code/text-to-motion/log/hu_GPRO_union15_3-11-16-53-51"
csv_filename = "/liujinxin/code/text-to-motion/log/hu_GPRO_union15_3-11-16-53-51/hu_GPRO_union15_3-11-16-53-51.csv"

# 正则表达式
pattern_section = re.compile(r"======== (.+)")
pattern_score = re.compile(r"Matching Score: ([\d.]+)")
pattern_fid = re.compile(r"FID: ([\d.]+)")

results = []

# 遍历日志文件
for filename in os.listdir(log_dir):
    log_path = os.path.join(log_dir, filename)

    # 只处理 .log 文件
    if not filename.endswith(".log"):
        continue

    with open(log_path, "r", encoding="utf-8") as file:
        current_section = ""
        matching_score = fid = None

        for line in file:
            match_section = pattern_section.match(line)
            if match_section:
                current_section = match_section.group(1)
                continue

            match_score = pattern_score.search(line)
            match_fid = pattern_fid.search(line)

            if match_score:
                matching_score = float(match_score.group(1))
            if match_fid:
                fid = float(match_fid.group(1))
                results.append([current_section, matching_score, fid])

# 写入CSV文件
csv_path = os.path.join(log_dir, csv_filename)
with open(csv_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Section", "Matching Score", "FID"])
    writer.writerows(results)

print(f"CSV文件已保存到 {csv_path}")
