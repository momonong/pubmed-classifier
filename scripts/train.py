import os
import json
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
from tensorboardX import SummaryWriter

# ======= Config =======
MODEL_CKPT = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
TOTAL_EPOCHS = 4
BEST_MODEL_DIR = "model/biobert"
LOG_DIR = "logs/manual"
SEED = 42

# ======= Fix seed =======
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ======= Load data =======
train_df = pd.read_csv("data/train_preprocessed.csv")
test_df = pd.read_csv("data/test_preprocessed.csv")
train_ds = Dataset.from_pandas(train_df)
test_ds = Dataset.from_pandas(test_df)

# ======= Tokenize =======
tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)
train_ds = train_ds.remove_columns(["text"])
test_ds = test_ds.remove_columns(["text"])
train_ds.set_format("torch")
test_ds.set_format("torch")

# ======= Model & Metrics =======
model = AutoModelForSequenceClassification.from_pretrained(MODEL_CKPT, num_labels=2)

def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

# ======= Trainer & Args =======
training_args = TrainingArguments(
    output_dir="./outputs",
    num_train_epochs=1,  # loop controlled manually
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir=LOG_DIR,
    logging_steps=10,
    save_strategy="no",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ======= Training Loop =======
writer = SummaryWriter(log_dir=LOG_DIR)
train_loss_log, eval_loss_log, eval_acc_log = [], [], []
best_eval_acc = 0.0

try:
    for epoch in range(TOTAL_EPOCHS):
        print(f"\n=== Epoch {epoch + 1} ===")
        trainer.train()

        train_loss = next((log["loss"] for log in reversed(trainer.state.log_history) if "loss" in log), None)
        eval_metrics = trainer.evaluate()
        eval_loss = eval_metrics["eval_loss"]
        eval_acc = eval_metrics["eval_accuracy"]

        # Logging
        train_loss_log.append(train_loss)
        eval_loss_log.append(eval_loss)
        eval_acc_log.append(eval_acc)
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/eval", eval_loss, epoch)
        writer.add_scalar("accuracy/eval", eval_acc, epoch)

        print(f"Epoch {epoch+1} - train_loss: {train_loss:.4f}, eval_loss: {eval_loss:.4f}, eval_acc: {eval_acc:.4f}")

        # Save best
        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            print(f"🌟 Saving new best model (acc={eval_acc:.4f}) to {BEST_MODEL_DIR}")
            model.save_pretrained(BEST_MODEL_DIR)
            tokenizer.save_pretrained(BEST_MODEL_DIR)
finally:
    writer.close()

# ======= Save training logs =======
os.makedirs("logs", exist_ok=True)
with open("logs/metrics_log.json", "w") as f:
    json.dump({
        "train_loss": train_loss_log,
        "eval_loss": eval_loss_log,
        "eval_acc": eval_acc_log
    }, f, indent=2)

# ======= Final Evaluation =======
preds = trainer.predict(test_ds).predictions.argmax(axis=1)
y_true = test_ds["label"]

report = classification_report(y_true, preds, target_names=["T0", "T2–T4"], output_dict=True)
print("\n📋 classification_report:")
print(classification_report(y_true, preds, target_names=["T0", "T2–T4"]))
pd.DataFrame(report).transpose().to_csv("logs/classification_report.csv")
