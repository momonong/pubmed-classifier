import scispacy
import spacy
from scispacy.linking import UmlsEntityLinker
import pandas as pd
from tqdm import tqdm

# 載入 SciSpacy 模型與 linker
nlp = spacy.load("en_core_sci_scibert")
linker = UmlsEntityLinker(resolve_abbreviations=True, name="umls")

nlp.add_pipe("scispacy_linker", config={"linker_name": "umls"})

# 載入資料
df = pd.read_csv("data/train_preprocessed.csv")  # or test_preprocessed.csv
texts = df["text"].tolist()

# 擷取每筆文本的 UMLS CUI
def extract_umls_cuis(text):
    doc = nlp(text)
    cuis = []
    for entity in doc.ents:
        for umls_ent in entity._.umls_ents:
            cui = umls_ent[0]
            cuis.append(cui)
    return list(set(cuis))  # 去重

df["umls_cuis"] = [extract_umls_cuis(text) for text in tqdm(texts)]
df.to_csv("data/train_with_umls.csv", index=False)
