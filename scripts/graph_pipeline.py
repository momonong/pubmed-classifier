import os
import math
import pickle
import networkx as nx
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.load_spacy import load_spacy_model
from metapub import PubMedFetcher, PubMedCentralFetcher
import warnings

warnings.filterwarnings("ignore")

nlp = load_spacy_model()


# --- TF-IDF corpus builder ---
class CorpusStats:
    def __init__(self):
        self.doc_count = 0
        self.term_doc_freq = defaultdict(int)
        self.term_pair_count = defaultdict(int)
        self.term_count = defaultdict(int)

    def add_document(self, text):
        self.doc_count += 1
        terms = set(extract_concepts(text))
        for term in terms:
            self.term_doc_freq[term] += 1
        for t1 in terms:
            self.term_count[t1] += 1
            for t2 in terms:
                if t1 != t2:
                    self.term_pair_count[(t1, t2)] += 1

    def pmi(self, t1, t2):
        pair = (t1, t2)
        if pair not in self.term_pair_count:
            return 0
        p_xy = self.term_pair_count[pair] / self.doc_count
        p_x = self.term_doc_freq[t1] / self.doc_count
        p_y = self.term_doc_freq[t2] / self.doc_count
        return math.log(p_xy / (p_x * p_y) + 1e-8)


# --- NLP utilities ---
def extract_concepts(text):
    doc = nlp(text)
    return list(set(chunk.text.strip().lower() for chunk in doc.noun_chunks))


# --- PubMed access ---
def fetch_pubmed_article(pmid: str):
    pmc_fetcher = PubMedCentralFetcher()
    pmid_fetcher = PubMedFetcher()
    article = pmid_fetcher.article_by_pmid(pmid)

    # 嘗試從 PMC 擷取全文
    try:
        pmcid = article.pmc
        full_article = pmc_fetcher.article_by_pmcid(pmcid)
        full_text = full_article.body_text
    except Exception:
        print(f"[!] Warning: No full text found for {pmid}. Fallback to abstract.")
        full_text = article.abstract

    return {
        "pmid": article.pmid,
        "title": article.title,
        "abstract": article.abstract,
        "full_text": full_text,
    }


# --- 建構 PMI 語意圖 ---
def build_pmi_graph(paper, corpus_stats: CorpusStats, pmi_threshold=1.0):
    G = nx.Graph()
    concepts = extract_concepts(paper["title"] + " " + paper["abstract"])

    for concept in concepts:
        G.add_node(concept)

    for i in range(len(concepts)):
        for j in range(i + 1, len(concepts)):
            t1, t2 = concepts[i], concepts[j]
            score = corpus_stats.pmi(t1, t2)
            if score >= pmi_threshold:
                G.add_edge(t1, t2, weight=score, type="pmi")

    return G


# --- 儲存圖檔與節點特徵 ---
def save_graph_and_features(G, pmid: str, outdir="data/graphs_pmi"):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"{pmid}.gpickle"), "wb") as f:
        pickle.dump(G, f)


# --- 主流程 ---
def process_pmid(pmid, corpus_stats):
    paper = fetch_pubmed_article(pmid)
    G = build_pmi_graph(paper, corpus_stats, pmi_threshold=1.0)
    save_graph_and_features(G, pmid)
    print(f"[✓] Graph built for PMID {pmid} with {len(G.nodes)} nodes, {len(G.edges)} edges.")


# --- 測試執行 ---
if __name__ == "__main__":
    test_pmid = "23210975"
    print("→ Building corpus statistics...")
    stats = CorpusStats()

    # 🧪 模擬語料庫：你可以在這裡加入多篇 abstract
    pmids = ["23210975"]
    for pmid in pmids:
        paper = fetch_pubmed_article(pmid)
        stats.add_document(paper["title"] + " " + paper["abstract"])

    print("→ Building PMI graph...")
    process_pmid(test_pmid, stats)
