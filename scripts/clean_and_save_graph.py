import os
import pickle
import networkx as nx
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# 來源與目標資料夾
DATA_ROOT = "data"
INPUT_DIRS = {
    "train": Path(DATA_ROOT) / "graphs_pmi" / "train",
    "test": Path(DATA_ROOT) / "graphs_pmi" / "test"
}
LABELS = {
    "train": pd.read_csv(Path(DATA_ROOT) / "labels" / "train_labels.csv", dtype=str),
    "test": pd.read_csv(Path(DATA_ROOT) / "labels" / "test_labels.csv", dtype=str)
}
OUTPUT_DIR = Path(DATA_ROOT) / "graphs_pmi_clean"

def clean_graphs(split):
    input_dir = INPUT_DIRS[split]
    label_df = LABELS[split]
    output_dir = OUTPUT_DIR / split
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_pmids = set(label_df["pmid"].astype(str))

    total, kept = 0, 0
    for path in tqdm(list(input_dir.glob("*.gpickle")), desc=f"Cleaning {split}"):
        pmid = path.stem
        total += 1

        if pmid not in valid_pmids:
            print(f"[跳過] 無對應 label: {pmid}")
            continue

        try:
            with open(path, "rb") as f:
                G = pickle.load(f)
            if not isinstance(G, nx.Graph):
                print(f"[跳過] 非 nx.Graph: {pmid}")
                continue
            if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
                print(f"[跳過] 空圖: {pmid}")
                continue

            # 強制所有邊都有 weight
            for u, v in G.edges():
                if "weight" not in G[u][v]:
                    G[u][v]["weight"] = 1

            # 儲存乾淨圖
            out_path = output_dir / f"{pmid}.gpickle"
            with open(out_path, "wb") as f:
                pickle.dump(G, f)
            kept += 1

        except Exception as e:
            print(f"[錯誤] 讀取失敗 {pmid}: {e}")

    print(f"\n✅ {split} 資料清理完成：保留 {kept}/{total} 張圖")

if __name__ == "__main__":
    clean_graphs("train")
    clean_graphs("test")
