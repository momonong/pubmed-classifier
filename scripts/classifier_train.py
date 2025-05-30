import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import classification_report
from tqdm import tqdm

# === 1. 載入訓練集的 [CLS] 向量與 label ===
X = torch.load("output/pubmedbert_cls.pt")  # 來自 PubMedBERT 的 [CLS] 向量
y = torch.load("output/labels.pt")          # 對應的 label

# === 2. 拆分 train/val（90%/10%）===
train_size = int(0.9 * len(X))
val_size = len(X) - train_size
train_ds, val_ds = random_split(TensorDataset(X, y), [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# === 3. 定義模型 ===
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

# === 4. 訓練 ===
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()
best_val_acc = 0

for epoch in range(10):
    model.train()
    for xb, yb in tqdm(train_loader):
        xb, yb = xb.to(device), yb.to(device)
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # 驗證
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    val_acc = correct / total
    print(f"Epoch {epoch+1}: val_acc = {val_acc:.4f}")

    # 儲存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "model/mlp_model.pt")
        print("✅ Saved best model!")

