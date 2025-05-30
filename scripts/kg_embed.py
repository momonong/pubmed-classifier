import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from datasets import Dataset
from tqdm import tqdm


# === 1. 路徑與裝置 ===
model_ckpt = "model"  # 儲存過的 fine-tuned encoder
data_path = "data/test_preprocessed.csv"  # 或 test_preprocessed.csv
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 2. 載入 tokenizer & model（只有 encoder 本體）===
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt).to(device)
model.eval()

# === 3. 載入與 tokenize 資料 ===
df = pd.read_csv(data_path)
dataset = Dataset.from_pandas(df)


def tokenize(batch):
    return tokenizer(
        batch["text"], truncation=True, padding="max_length", max_length=512
    )


dataset = dataset.map(tokenize, batched=True)
dataset = dataset.remove_columns(["text"])
dataset.set_format("torch")

# === 4. 建立 DataLoader ===
loader = DataLoader(dataset, batch_size=16)

# === 5. 推論 CLS 向量 ===
all_cls_embeddings = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(loader, desc="Extracting [CLS] embeddings"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeds = outputs.last_hidden_state[:, 0, :]  # 取 [CLS] 向量

        all_cls_embeddings.append(cls_embeds.cpu())
        all_labels.append(batch["label"])

# === 6. 儲存結果 ===
cls_embeddings = torch.cat(all_cls_embeddings, dim=0)  # [N, hidden_dim]
labels = torch.cat(all_labels, dim=0)

torch.save(cls_embeddings, os.path.join(output_dir, "test_pubmedbert_cls.pt"))
torch.save(labels, os.path.join(output_dir, "test_labels.pt"))
