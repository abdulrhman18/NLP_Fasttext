import pandas as pd
import re
import nltk
from gensim.models.fasttext import FastText
from gensim.models.fasttext import load_facebook_model
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")

# Define stopwords and lemmatizer
en_stop = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Load Yelp Dataset (Use only 'text' column)
file_path = r"C:\Users\eng abdulrhman\Downloads\yelp_academic_dataset_tip.json (1)\yelp_academic_dataset_tip.json"
yelp_data = pd.read_json(file_path, lines=True)
texts = yelp_data["text"].dropna().tolist()


def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove numbers and special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces

    words = text.split()  # Word tokenization
    words = [lemmatizer.lemmatize(word) for word in words if len(word) > 3 and word not in en_stop]
    return words  # Return list of words (FastText expects list format)

# Apply preprocessing
processed_texts = [preprocess_text(text) for text in texts]
print(processed_texts)

# Step 1: Train FastText Model on Yelp Data
fasttext_model = FastText(vector_size=100, window=5, min_count=5, workers=4, sg=1, epochs=10)
fasttext_model.build_vocab(corpus_iterable=processed_texts)
fasttext_model.train(corpus_iterable=processed_texts, total_examples=len(processed_texts), epochs=10)
fasttext_model.save("yelp_fasttext.model")
print("Model trained and saved successfully.")


def test_model(model, word):
    try:
        print(f"\nTop 10 similar words to '{word}':")
        print(model.wv.most_similar(word, topn=10))

        print(f"\nTop 10 opposite words to '{word}':")
        print(model.wv.most_similar(negative=[word], topn=10))
    except KeyError:
        print(f"Word '{word}' not in vocabulary!")

# Test words WITH our Model
test_words = ["mother", "good"]
for word in test_words:
    test_model(fasttext_model, word)

# Step 2: Load Pretrained FastText Model and Test
pretrained_model = load_facebook_model(r"C:\Users\eng abdulrhman\Desktop\NLP ASSIGNMENT\cc.en.300.bin")
print("Pretrained FastText model loaded successfully!")

# Test With pretrained Model
for word in test_words:
    test_model(pretrained_model, word)

# Step 3: Update Pretrained Model with Yelp Data
pretrained_model.build_vocab(corpus_iterable=processed_texts, update=True)
pretrained_model.train(corpus_iterable=processed_texts, total_examples=len(processed_texts), epochs=20)
pretrained_model.save("updated_fasttext_yelp.model")
print("Updated FastText model saved successfully!")

# Test after Update Pretrained model
for word in test_words:
    test_model(pretrained_model, word)