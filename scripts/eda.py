import os
import pandas as pd
import matplotlib.pyplot as plt

# 設定路徑
excel_path = "data/Assignment 2 Dataset.xlsx"
csv_train_path = "data/train.csv"
csv_test_path = "data/test.csv"

# 建立 figures 資料夾
os.makedirs("figures", exist_ok=True)

def analyze_and_plot(df: pd.DataFrame, name: str, label_col: str):
    """進行 label 分析與繪圖"""
    print(f"\n📊 分析：{name}")
    label_counts = df[label_col].value_counts().sort_index()
    label_pct = df[label_col].value_counts(normalize=True).sort_index() * 100

    print("各類別文章數量：\n", label_counts)
    print("各類別比例（百分比）：\n", label_pct.round(2))

    # 長條圖
    plt.figure()
    label_counts.plot(kind='bar', title=f"{name} - Label Distribution")
    plt.xticks(ticks=[0, 1], labels=["T0", "T2–T4"], rotation=0)
    plt.ylabel("Number of Articles")
    plt.tight_layout()
    plt.savefig(f"figures/{name}_label_distribution.png")
    plt.close()

# ---------- 分析 Excel 中的兩個 sheet ----------
if os.path.exists(excel_path):
    df_sheets = pd.read_excel(excel_path, sheet_name=None, engine="openpyxl")

    for sheet_name, df in df_sheets.items():
        label_col = df.columns[0]
        analyze_and_plot(df, name=sheet_name, label_col=label_col)
else:
    print(f"⚠️ Excel 檔案不存在：{excel_path}")

# ---------- 分析 train.csv ----------
if os.path.exists(csv_train_path):
    df_train = pd.read_csv(csv_train_path)
    label_col = df_train.columns[0]
    analyze_and_plot(df_train, name="train_csv", label_col=label_col)

# ---------- 分析 test.csv ----------
if os.path.exists(csv_test_path):
    df_test = pd.read_csv(csv_test_path)
    label_col = df_test.columns[0]
    analyze_and_plot(df_test, name="test_csv", label_col=label_col)
