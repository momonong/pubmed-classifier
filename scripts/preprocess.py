import pandas as pd
import os

# 設定路徑
train_path = "data/train.csv"
test_path = "data/test.csv"
output_train_path = "data/train_preprocessed.csv"
output_test_path = "data/test_preprocessed.csv"

# 建立資料夾（如果不存在）
os.makedirs("data", exist_ok=True)

def preprocess(file_path, output_path):
    df = pd.read_csv(file_path)

    # 嘗試自動偵測欄位
    title_col = next((col for col in df.columns if "title" in col.lower()), None)
    abstract_col = next((col for col in df.columns if "abstract" in col.lower()), None)
    label_col = next((col for col in df.columns if "label" in col.lower() or "curate" in col.lower()), None)

    if not all([title_col, abstract_col, label_col]):
        print(f"⚠️ 欄位偵測失敗，請確認 {file_path} 中有 title、abstract、label 三欄")
        return

    # 合併成單一 text 欄位（用句號分隔）
    df["text"] = df[title_col].fillna('') + ". " + df[abstract_col].fillna('')

    # 只保留 text 和 label 欄位
    df = df[["text", label_col]]
    df.rename(columns={label_col: "label"}, inplace=True)

    # 移除空的文字或 label 欄位
    df = df[df["text"].str.strip().astype(bool)]
    df = df[df["label"].notnull()]

    df.to_csv(output_path, index=False)
    print(f"✅ 已處理並儲存：{output_path}（共 {len(df)} 筆）")

if __name__ == "__main__":
    preprocess(train_path, output_train_path)
    preprocess(test_path, output_test_path)
