"""
从 美国土壤类型.xlsx 生成州级静态特征映射表。
与 Product_model/new_Data/short_seq_nonorm.ipynb 的处理逻辑一致：
  - 土壤平均有机碳含量（%） → 四分位数分桶 → carbon_bucket
  - 土壤平均PH             → 自定义分箱     → ph_bucket

输出：
  - us_state_soil.csv       : 州级土壤映射表 (State, carbon_bucket, ph_bucket)
  - us_carbon_bucket_id.json: carbon_bucket → ID 映射
  - us_ph_bucket_id.json    : ph_bucket → ID 映射
"""

import os
import json
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = SCRIPT_DIR

# 引用 Product_model 下的原始 Excel
SRC_EXCEL = r"C:\Users\colde\Desktop\model_crucial\Product_model\new_Data\美国土壤类型.xlsx"

# ============================================================
# 1. 读取
# ============================================================
df = pd.read_excel(SRC_EXCEL)
print(f"原始行数: {len(df)}")
print(f"列名: {list(df.columns)}")

# ============================================================
# 2. 分桶（同 notebook）
# ============================================================

# 碳含量四分位数分桶
df["carbon_bucket"] = pd.qcut(
    df["土壤平均有机碳含量（%）"],
    q=4,
    labels=["极低碳", "低碳", "中碳", "高碳"]
)

# pH 自定义分箱
df["ph_bucket"] = pd.cut(
    df["土壤平均PH"],
    bins=[0, 6.0, 7.5, 14],
    labels=["酸性", "中性", "碱性"]
)

# ============================================================
# 3. 州名标准化（同 notebook）
# ============================================================
df["State"] = df["州名称"].str.strip().str.lower().str.replace(" ", "_")

# 只保留有用列
df_out = df[["State", "carbon_bucket", "ph_bucket"]].copy()

# 去除非美国本土行（如 Puerto Rico, Guam 等）
# 保留和 notebook 中 state list 以及 cropnet 数据集有关的州
print(f"\n分桶统计:")
print(f"  carbon_bucket: {df_out['carbon_bucket'].value_counts().to_dict()}")
print(f"  ph_bucket:     {df_out['ph_bucket'].value_counts().to_dict()}")

# ============================================================
# 4. 输出 CSV 映射表
# ============================================================
csv_path = os.path.join(OUTPUT_DIR, "us_state_soil.csv")
df_out.to_csv(csv_path, index=False, encoding="utf-8")
print(f"\n✓ 已保存: {csv_path}")

# ============================================================
# 5. 输出 ID 映射 JSON（同 notebook）
# ============================================================
carbon_id = {s: idx for idx, s in enumerate(df_out["carbon_bucket"].unique())}
ph_id = {s: idx for idx, s in enumerate(df_out["ph_bucket"].unique())}

print(f"\n  carbon_bucket → ID: {carbon_id}")
print(f"  ph_bucket → ID:     {ph_id}")

with open(os.path.join(OUTPUT_DIR, "us_carbon_bucket_id.json"), "w", encoding="utf-8") as f:
    json.dump(carbon_id, f, ensure_ascii=False, indent=2)

with open(os.path.join(OUTPUT_DIR, "us_ph_bucket_id.json"), "w", encoding="utf-8") as f:
    json.dump(ph_id, f, ensure_ascii=False, indent=2)

print(f"\n✓ ID 映射已保存到: {OUTPUT_DIR}")
print(f"\n预览 (前5行):")
print(df_out.head().to_string())
