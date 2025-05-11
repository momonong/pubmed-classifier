import os
import glob
import pickle
import torch
import pandas as pd
import networkx as nx
from tqdm import tqdm
from pathlib import Path
from sentence_transformers import SentenceTransformer
from torch.nn import Linear
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import random

# 固定 seed
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)

# ====== Dataset class ======
class SciBERTGraphDataset(InMemoryDataset):
    def __init__(self, graph_dir, label_path, encoder):
        self.graph_dir = graph_dir
        self.label_path = label_path
        self.encoder = encoder
        super().__init__(None, None)
        self.data_list = self.load_data()

    def load_data(self):
        label_df = pd.read_csv(self.label_path, dtype={"pmid": str}).set_index("pmid")
        files = sorted(glob.glob(os.path.join(self.graph_dir, "*.gpickle")))
        data_list = []

        for path in tqdm(files, desc="Loading test graphs"):
            pmid = Path(path).stem
            if pmid not in label_df.index:
                print(f"[跳過] 無對應 label: {pmid}")
                continue

            with open(path, "rb") as f:
                G = pickle.load(f)
            if not isinstance(G, nx.Graph) or len(G.nodes) == 0 or len(G.edges) == 0:
                print(f"[跳過] 無效圖（非 nx.Graph 或空圖）: {pmid}")
                continue

            node_texts = list(G.nodes)
            if not any(node.strip() for node in node_texts):
                print(f"[跳過] 空節點名稱: {pmid}")
                continue

            X = self.encoder.encode(node_texts, show_progress_bar=False)

            for u, v, data in G.edges(data=True):
                if "weight" not in data:
                    data["weight"] = 1.0

            pyg = from_networkx(G)
            pyg.x = torch.tensor(X, dtype=torch.float)
            pyg.y = torch.tensor([label_df.loc[pmid, "label"]], dtype=torch.long)
            data_list.append(pyg)

        print(f"✅ 成功載入 {len(data_list)} 個圖（共 {len(files)}）")
        return data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


# ====== GAT 模型定義 ======
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64, heads=4, dropout=0.5):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATConv(
            hidden_channels * heads,
            hidden_channels,
            heads=1,
            concat=False,
            dropout=dropout,
        )
        self.lin = Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch):
        if edge_index.size(1) == 0:
            return torch.zeros((x.size(0), self.lin.out_features), device=x.device)
        x = self.gat1(x, edge_index).relu()
        x = self.gat2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.lin(x)


# ====== 評估流程 ======
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            y_true += batch.y.tolist()
            y_pred += pred.tolist()
    return y_true, y_pred


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥 Using device: {device}")

    encoder = SentenceTransformer("allenai/scibert_scivocab_uncased")
    test_set = SciBERTGraphDataset(
        "data/graphs_pmi_clean/test", "data/labels/test_labels.csv", encoder
    )
    test_loader = DataLoader(test_set, batch_size=32)

    in_dim = test_set[0].x.shape[1]
    model = GAT(in_channels=in_dim, hidden_channels=128, heads=8).to(device)
    model_path = "models/gat_best_overall.pt"
    assert os.path.exists(model_path), f"❌ 模型不存在：{model_path}"
    model.load_state_dict(torch.load(model_path))
    print(f"✅ 模型載入完成：{model_path}")

    y_true, y_pred = evaluate(model, test_loader, device)

    print("\n🎯 測試結果:")
    print(f"📈 Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, digits=4))


if __name__ == "__main__":
    main()
