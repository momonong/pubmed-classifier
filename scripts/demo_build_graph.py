import networkx as nx
from itertools import combinations
from metapub import PubMedFetcher
from utils.load_spacy import load_spacy_model

# 載入 NLP 模型（建議使用 scispacy，如 en_core_sci_sm）

nlp = load_spacy_model()


def fetch_pubmed_article(pmid: str):
    fetcher = PubMedFetcher()
    article = fetcher.article_by_pmid(pmid)
    return {
        "pmid": article.pmid,
        "title": article.title,
        "abstract": article.abstract,
        "authors": article.authors,
        "journal": article.journal,
        "mesh": getattr(article, "mesh_headings", []),
        "keywords": getattr(article, "keywords", []),
    }


def extract_concepts(text):
    doc = nlp(text)
    return list(set(chunk.text.strip().lower() for chunk in doc.noun_chunks))


def build_semantic_graph(title, abstract, mesh_terms=None, keywords=None):
    G = nx.Graph()
    sentences = list(nlp(abstract).sents)
    all_concepts = extract_concepts(title + " " + abstract)

    if mesh_terms:
        all_concepts += [term.lower() for term in mesh_terms]
    if keywords:
        all_concepts += [kw.lower() for kw in keywords]

    for concept in all_concepts:
        G.add_node(concept)

    for sent in sentences:
        sent_concepts = extract_concepts(sent.text)
        for u, v in combinations(set(sent_concepts), 2):
            if G.has_edge(u, v):
                G[u][v]["weight"] += 1
            else:
                G.add_edge(u, v, type="co_occurrence", weight=1)

    return G


def print_graph(G):
    print(f"Nodes:{len(G.nodes)}")
    for node in G.nodes:
        print(f"- {node}")
    print(f"\nEdges: {len(G.edges)}")
    for u, v, data in G.edges(data=True):
        print(f"- ({u}) --[{data['type']}, w={data['weight']}]--> ({v})")


if __name__ == "__main__":
    # ==== 測試第一篇文章：PMID 23210975 ====
    paper = fetch_pubmed_article("27834361")
    print("PMID:", paper["pmid"])
    print("Title:", paper["title"])
    print("\nAbstract:\n", paper["abstract"])

    G = build_semantic_graph(
        title=paper["title"],
        abstract=paper["abstract"],
        mesh_terms=paper["mesh"],
        keywords=paper["keywords"],
    )
    print("\n================ Semantic Graph ================")
    print_graph(G)
