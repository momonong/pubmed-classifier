# scripts/build_train_graphs.py
import os, pickle, math
import pandas as pd
import networkx as nx
from collections import defaultdict
from metapub import PubMedFetcher
from utils.load_spacy import load_spacy_model
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

nlp = load_spacy_model()
fetcher = PubMedFetcher()

GRAPH_DIR = "data/graphs_pmi/train"
LABEL_PATH = "data/labels/train_labels.csv"
PMI_MODEL_PATH = "data/pmi_model/train_corpus.pkl"
EXCEL_PATH = "data/Assignment 2 Dataset.xlsx"
PMI_THRESHOLD = 0.5

os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LABEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(PMI_MODEL_PATH), exist_ok=True)


class CorpusStats:
    def __init__(self):
        self.doc_count = 0
        self.term_doc_freq = defaultdict(int)
        self.term_pair_count = defaultdict(int)

    def add_document(self, text):
        self.doc_count += 1
        terms = set(extract_concepts(text))
        for t1 in terms:
            self.term_doc_freq[t1] += 1
            for t2 in terms:
                if t1 != t2:
                    self.term_pair_count[(t1, t2)] += 1

    def pmi(self, t1, t2):
        if (t1, t2) not in self.term_pair_count:
            return 0
        p_xy = self.term_pair_count[(t1, t2)] / self.doc_count
        p_x = self.term_doc_freq[t1] / self.doc_count
        p_y = self.term_doc_freq[t2] / self.doc_count
        return math.log(p_xy / (p_x * p_y) + 1e-8)


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
    df = pd.read_excel(EXCEL_PATH, sheet_name="trainset_2286")
    stats = CorpusStats()
    labels = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building PMI Stats"):
        text = fetch_article_text(row["PMID"])
        if text:
            stats.add_document(text)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building Graphs"):
        pmid, label = str(row["PMID"]), int(row["Curate (0: T0, 1: T2/4)"])
        text = fetch_article_text(pmid)
        if not text:
            continue
        concepts = extract_concepts(text)
        G = build_graph(concepts, stats)
        with open(os.path.join(GRAPH_DIR, f"{pmid}.gpickle"), "wb") as f:
            pickle.dump(G, f)
        labels.append({"pmid": pmid, "label": label})

    with open(PMI_MODEL_PATH, "wb") as f:
        pickle.dump(stats, f)
    pd.DataFrame(labels).to_csv(LABEL_PATH, index=False)
    print("✅ Train graphs and PMI model saved.")


if __name__ == "__main__":
    main()
