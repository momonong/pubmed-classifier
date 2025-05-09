# utils/load_spacy.py
import spacy


def load_spacy_model():
    """
    Tries to load SciSpaCy model (en_core_sci_sm),
    falls back to en_core_web_sm if not available.
    """
    try:
        return spacy.load("en_core_sci_sm")
    except OSError:
        print("[Warning] en_core_sci_sm not available, falling back to en_core_web_sm")
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError("Neither en_core_sci_sm nor en_core_web_sm is installed. Please install one of them.")
