import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report

# === 1. 載入 testing set 的 [CLS] 向量與對應 label ===
X = torch.load("output/test_pubmedbert_cls.pt")  # 來自 PubMedBERT 的 [CLS] 向量
y = torch.load("output/test_labels.pt")          # 對應的 label

# === 2. 用 DataLoader 包裝 ===
test_ds = TensorDataset(X, y)
test_loader = DataLoader(test_ds, batch_size=32)

# === 3. 定義與訓練一致的 MLP 架構 ===
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLPClassifier(input_dim=X.shape[1]).to(device)

# 載入你訓練好的分類器權重（如果你有保存）
model.load_state_dict(torch.load("model/mlp_model.pt"))

# === 4. 推論 ===
model.eval()
all_preds = []
with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(device)
        logits = model(xb)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())

# === 5. 顯示分類報告 ===
print("\n📋 classification_report:")
print(classification_report(y.cpu(), all_preds, target_names=["T0", "T2–T4"]))
