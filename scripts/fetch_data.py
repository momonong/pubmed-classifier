import pandas as pd
from metapub import PubMedFetcher
import time
import os

fetcher = PubMedFetcher()

def fetch_metadata(pmid):
    try:
        article = fetcher.article_by_pmid(str(pmid))
        print(pmid, "fetch success")
        return {
            "PMID": pmid,
            "Title": article.title or "",
            "Abstract": article.abstract or ""
        }
    except Exception as e:
        # print(f"❌ Error fetching PMID {pmid}: {e}")
        print(pmid, "fetch failed")
        return {
            "PMID": pmid,
            "Title": "",
            "Abstract": ""
        }

def process_sheet(file_path, sheet_name, output_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")
    
    # 自動找出 Label 和 PMID 欄
    label_col = df.columns[0]
    pmid_col = df.columns[1]
    
    print(f"🚀 處理 sheet：{sheet_name}，共 {len(df)} 筆資料")

    records = []
    for i, row in df.iterrows():
        label = row[label_col]
        pmid = row[pmid_col]
        info = fetch_metadata(pmid)
        info["Label"] = label
        records.append(info)
        # time.sleep(0.34)  # 避免 API 過載

    result_df = pd.DataFrame(records)
    result_df = result_df[result_df["Abstract"].str.strip().astype(bool)]  # 過濾空摘要
    result_df.to_csv(output_name, index=False)
    print(f"✅ 已儲存 {output_name}，有效資料數：{len(result_df)}")

if __name__ == "__main__":
    excel_path = "data/Assignment 2 Dataset.xlsx"  # 請替換為你的檔名

    process_sheet(excel_path, sheet_name="trainset_2286", output_name="data/train.csv")
    process_sheet(excel_path, sheet_name="testset_400", output_name="data/test.csv")
