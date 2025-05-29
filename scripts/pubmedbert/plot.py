import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import Dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorboard.backend.event_processing import event_accumulator

# === 載入模型與 tokenizer ===
model = AutoModelForSequenceClassification.from_pretrained("model/biobert")
tokenizer = AutoTokenizer.from_pretrained("model/biobert")

# === 載入 test 資料並預處理 ===
test_df = pd.read_csv("data/test_preprocessed.csv")
test_ds = Dataset.from_pandas(test_df)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)

test_ds = test_ds.map(tokenize, batched=True)
test_ds = test_ds.remove_columns(["text"])
test_ds.set_format("torch")

# === 預測 ===
trainer = Trainer(model=model, tokenizer=tokenizer)
raw_pred = trainer.predict(test_ds)
preds = np.argmax(raw_pred.predictions, axis=1)
y_true = test_ds["label"]

# === 畫 Confusion Matrix ===
os.makedirs("figures", exist_ok=True)
labels = ["T0", "T2–T4"]
cm = confusion_matrix(y_true, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("figures/confusion_matrix.png")
plt.close()

# === 從 TensorBoard log 畫 Loss & Accuracy 曲線 ===
log_dir = "logs/manual"
event_file = next((os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.startswith("events.out")), None)

if event_file:
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()

    def plot_scalar(tag, ylabel, filename):
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]

        plt.figure()
        plt.plot(steps, values, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(ylabel + " Curve")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"figures/{filename}")
        plt.close()

    plot_scalar("loss/train", "Training Loss", "loss_curve.png")
    plot_scalar("loss/eval", "Validation Loss", "eval_loss_curve.png")
    plot_scalar("accuracy/eval", "Validation Accuracy", "accuracy_curve.png")
else:
    print("⚠️ 找不到 TensorBoard log 檔案，無法繪製 loss / accuracy 圖。")
