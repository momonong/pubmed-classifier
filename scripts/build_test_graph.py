# scripts/build_test_graphs.py
import os, pickle
import pandas as pd
import networkx as nx
from tqdm import tqdm
from metapub import PubMedFetcher
from utils.load_spacy import load_spacy_model
from utils.pmi import CorpusStats
import warnings

warnings.filterwarnings("ignore")

nlp = load_spacy_model()
fetcher = PubMedFetcher()

EXCEL_PATH = "data/Assignment 2 Dataset.xlsx"
PMI_MODEL_PATH = "data/pmi_model/train_corpus.pkl"
GRAPH_DIR = "data/graphs_pmi/test"
LABEL_PATH = "data/labels/test_labels.csv"
PMI_THRESHOLD = 0.5

os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LABEL_PATH), exist_ok=True)


def extract_concepts(text):
    doc = nlp(text)
    return list(set(chunk.text.strip().lower() for chunk in doc.noun_chunks))


def fetch_article_text(pmid):
    try:
        article = fetcher.article_by_pmid(str(pmid))
        title = article.title or ""
        abstract = article.abstract or ""
        if not (title or abstract):
            raise ValueError("Empty title & abstract")
        return title + " " + abstract
    except Exception as e:
        print(f"[!] Failed to fetch PMID {pmid}: {e}")
        return None


def build_graph(concepts, stats, threshold=PMI_THRESHOLD):
    G = nx.Graph()
    for c in concepts:
        G.add_node(c)
    for i in range(len(concepts)):
        for j in range(i + 1, len(concepts)):
            t1, t2 = concepts[i], concepts[j]
            score = stats.pmi(t1, t2)
            if score >= threshold:
                G.add_edge(t1, t2, weight=score, type="pmi")
    return G


def main():
    df = pd.read_excel(EXCEL_PATH, sheet_name="testset_400")
    with open(PMI_MODEL_PATH, "rb") as f:
        stats = pickle.load(f)

    labels = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building Test Graphs"):
        pmid, label = str(row["PMID"]), int(row["Curate (0: T0, 1: T2/4)"])
        text = fetch_article_text(pmid)
        if not text:
            continue
        concepts = extract_concepts(text)
        G = build_graph(concepts, stats)
        with open(os.path.join(GRAPH_DIR, f"{pmid}.gpickle"), "wb") as f:
            pickle.dump(G, f)
        labels.append({"pmid": pmid, "label": label})

    pd.DataFrame(labels).to_csv(LABEL_PATH, index=False)
    print("✅ Test graphs built using train PMI model.")


if __name__ == "__main__":
    main()
