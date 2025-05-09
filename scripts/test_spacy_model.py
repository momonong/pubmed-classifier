import spacy

nlp = spacy.load("en_core_sci_sm")
doc = nlp("Alterations in the hypocretin receptor 2 and preprohypocretin genes produce narcolepsy in some animals.")
for token in doc:
    print(token.text, token.lemma_, token.pos_)
