"""Pre-sprint check: do the two models load and give sensible outputs?"""
from sentence_transformers import CrossEncoder, SentenceTransformer, util

premise = "The treaty was signed on February 7, 1945, in Yalta."
claims = [
    "The treaty was signed on February 7, 1945.",   # supported
    "The treaty was signed on February 7, 2022.",   # one-fact change
    "The treaty was not signed in Yalta.",          # negation
    "The treaty ended the war in Europe.",          # not in source
    "Yalta is where the treaty was signed.",        # paraphrase
]

nli = CrossEncoder("cross-encoder/nli-deberta-v3-base")
id2label = nli.model.config.id2label   # read label order from the model itself
labels = [id2label[i].lower() for i in range(len(id2label))]
print("label order:", labels)
emb = SentenceTransformer("BAAI/bge-base-en-v1.5")

p_vec = emb.encode(premise, normalize_embeddings=True)
for c in claims:
    scores = nli.predict([(premise, c)], apply_softmax=True)[0]
    cos = float(util.cos_sim(p_vec, emb.encode(c, normalize_embeddings=True)))
    print(f"{labels[scores.argmax()]:<13} cos={cos:.3f}  {c}")
