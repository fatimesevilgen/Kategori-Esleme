import joblib
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("xlm-r-bert-base-nli-stsb-mean-tokens") # Çok dilli, BERT tabanlı. Daha büyük ve güçlü.

clf = joblib.load("model.pkl")

def predict_category(user_input: str):
    user_input_embedded = model.encode([user_input])
    prediction = clf.predict(user_input_embedded)
    return prediction[0]

while True:
    user_input = input("\nEtkinlik kategorisi gir (çıkmak için 'q'): ")
    user_input = user_input.capitalize()
    if user_input.lower() == "q":
        break
    predicted_category = predict_category(user_input)
    print(f"Tahmin edilen kategori: {predicted_category}")