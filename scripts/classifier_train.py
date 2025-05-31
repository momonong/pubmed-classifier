import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm
import os
import random
import numpy as np

# === Set Seed ===
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(112)

# === Mixup 實作 ===
def mixup_data(x, y, alpha=1.0):
    if alpha <= 0:
        return x, y, None, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# === Focal Loss 實作 ===
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, input, target):
        ce_loss = self.ce(input, target)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss

# === Load Data ===
X_train = torch.load("output/pubmedbert_cls.pt")
y_train = torch.load("output/labels.pt")
X_test = torch.load("output/test_pubmedbert_cls.pt")
y_test = torch.load("output/test_labels.pt")
print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

# === 強化版 MLP ===
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 768, 256], dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[2], 2)
        )

    def forward(self, x):
        return self.net(x)

# === Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLPClassifier(input_dim=X_train.shape[1]).to(device)

class_weights = torch.tensor([1.0, 1.3], dtype=torch.float).to(device)
loss_fn = FocalLoss(gamma=2.0, weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

best_acc = 0
patience = 5
wait = 0
print("🔧 Start training...")

# === Training ===
for epoch in range(30):
    model.train()
    total_loss = 0
    for xb, yb in tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}"):
        xb, yb = xb.to(device), yb.to(device)

        # Mixup
        mixed_x, y_a, y_b, lam = mixup_data(xb, yb, alpha=0.8)
        out = model(mixed_x)
        loss = lam * loss_fn(out, y_a) + (1 - lam) * loss_fn(out, y_b)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()
    print(f"📉 Epoch {epoch+1}: Train Loss = {total_loss:.4f}")

    # === Save best model ===
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            y_true.extend(yb.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    acc = sum([y1 == y2 for y1, y2 in zip(y_true, y_pred)]) / len(y_true)
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "model/best_mlp.pt")
        print("✅ Saved best model!")
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("⏹️ Early stopping")
            break

# === Evaluation ===
print("🔍 Running on test set...")
model.load_state_dict(torch.load("model/best_mlp.pt"))
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb).argmax(dim=1)
        y_true.extend(yb.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

print("📊 [Test Set Results]")
print(classification_report(y_true, y_pred, digits=4))
