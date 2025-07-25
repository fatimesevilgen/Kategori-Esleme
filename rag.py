# import pandas as pd
# from sentence_transformers import SentenceTransformer, util
# import torch

# df = pd.read_csv("sample_event_data.csv")
# corpus_texts = df["text"].tolist()
# corpus_labels = df["label"].tolist()

# model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True)

# # Tahmin fonksiyonu
# def predict_by_similarity(user_input: str, top_k: int = 3):
#     user_embedding = model.encode(user_input, convert_to_tensor=True)
#     cosine_scores = util.cos_sim(user_embedding, corpus_embeddings)[0]

#     top_results = torch.topk(cosine_scores, k=top_k)
#     for score, idx in zip(top_results.values, top_results.indices):
#         print(f"🟢 '{corpus_texts[idx]}' → {corpus_labels[idx]} (score: {score:.4f})")

# # 5. Test
# while True:
#     user_input = input("\nEtkinlik açıklaması gir (çıkmak için 'q'): ")
#     if user_input.lower() == "q":
#         break
#     predict_by_similarity(user_input, top_k=5)


from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
from sklearn.utils import shuffle

# 1. Veriyi yükle ve karıştır
df = pd.read_csv("sample_event_data.csv")
df = shuffle(df, random_state=42).reset_index(drop=True)

corpus_texts = df["text"].tolist()
corpus_labels = df["label"].tolist()

# 2. Model ve embedding
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings = model.encode(corpus_texts, convert_to_tensor=True)

# 3. Doğruluk hesapla
correct = 0
total = len(corpus_texts)

for i in range(total):
    input_vec = embeddings[i]
    rest_vecs = torch.cat((embeddings[:i], embeddings[i+1:]))  # kendi örneği hariç tut
    rest_labels = corpus_labels[:i] + corpus_labels[i+1:]

    cosine_scores = util.cos_sim(input_vec, rest_vecs)[0]
    best_idx = cosine_scores.argmax().item()
    predicted_label = rest_labels[best_idx]
    true_label = corpus_labels[i]

    if predicted_label == true_label:
        correct += 1

accuracy = correct / total
print(f"🔎 Similarity tabanlı doğruluk: {accuracy:.2%}")
