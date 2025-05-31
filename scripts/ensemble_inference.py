import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

# === 強化版 MLP 結構 ===
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

# === 載入資料 ===
X_test = torch.load("output/test_pubmedbert_cls.pt")
y_test = torch.load("output/test_labels.pt")
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

# === 載入模型 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_paths = [
    "model/ensemble_mlp_0.pt",
    "model/ensemble_mlp_1.pt",
    "model/ensemble_mlp_2.pt"
]

models = []
for path in model_paths:
    model = MLPClassifier(input_dim=X_test.shape[1]).to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    models.append(model)

# === Soft Voting Ensemble 預測 ===
y_true, y_pred = [], []
all_probs = []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        probs = torch.zeros(xb.size(0), 2).to(device)
        for model in models:
            logits = model(xb)
            probs += torch.softmax(logits, dim=1)
        avg_probs = probs / len(models)
        preds = avg_probs.argmax(dim=1)
        y_true.extend(yb.tolist())
        y_pred.extend(preds.cpu().tolist())

# === 評估指標 ===
acc = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average='macro')
print(f"✅ Ensemble Accuracy: {acc:.4f}, Macro F1: {macro_f1:.4f}")

# === 輸出 Classification Report ===
report = classification_report(y_true, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()
os.makedirs("logs", exist_ok=True)
df_report.to_csv("logs/classification_report.csv")

# === 繪製 Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Ensemble Confusion Matrix")
plt.tight_layout()
plt.savefig("logs/confusion_matrix.png")
plt.close()
