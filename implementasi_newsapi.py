import numpy as np
import requests
from collections import Counter
import re


class SkipGramModel:
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # input dan hidden
        self.W1 = np.random.randn(vocab_size, embedding_dim)

        # hidden dan output
        self.W2 = np.random.randn(embedding_dim, vocab_size)

    def forward(self, one_hot_vector):
        hidden_layer = np.dot(one_hot_vector, self.W1)
        output_layer = np.dot(hidden_layer, self.W2)
        output_layer = self._softmax(output_layer)
        return hidden_layer, output_layer

    def backward(self, one_hot_vector, target_vector, learning_rate=0.01):
        hidden_layer, output_layer = self.forward(one_hot_vector)
        error = target_vector - output_layer

        # Compute Gradients
        output_layer_gradient = np.outer(hidden_layer, error)
        hidden_layer_gradient = np.outer(one_hot_vector, np.dot(self.W2, error))

        # Update Weights
        self.W1 -= learning_rate * hidden_layer_gradient
        self.W2 -= learning_rate * output_layer_gradient

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def get_most_similar(self, word, word2idx, top_n=5):
        if word not in word2idx:
            return []
        word_vector = self.W1[word2idx[word]]
        similarities = {}
        for other_word, idx in word2idx.items():
            if other_word == word:
                continue
            other_vector = self.W1[idx]
            similarity = np.dot(word_vector, other_vector) / (
                        np.linalg.norm(word_vector) * np.linalg.norm(other_vector))
            similarities[other_word] = similarity
        return sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_n]


def fetch_news(api_key, query="technology", page_size=10):
    url = f"https://newsapi.org/v2/everything?q={query}&pageSize={page_size}&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get("articles", [])
        texts = [article["title"] for article in articles if article["title"]]
        return texts
    else:
        print("Error fetching news:", response.status_code)
        return []


def preprocess_texts(texts):
    words = []
    for text in texts:
        text = text.lower()
        text = re.sub(r'[^a-z ]', '', text)
        words.extend(text.split())
    return words


def build_training_pairs(words, window_size):
    training_pairs = []
    for i, target_word in enumerate(words):
        for j in range(-window_size, window_size + 1):
            if j == 0 or i + j < 0 or i + j >= len(words):
                continue
            training_pairs.append((target_word, words[i + j]))
    return training_pairs


def train_example(api_key, window_sizes=[1, 2, 3], embedding_dims=[20, 50, 100]):
    news_titles = fetch_news(api_key)
    words = preprocess_texts(news_titles)
    if not words:
        print("No words found after preprocessing. Exiting.")
        return None, None

    word_counts = Counter(words)
    vocab = list(word_counts.keys())
    word2idx = {word: idx for idx, word in enumerate(vocab)}

    for window_size in window_sizes:
        for embedding_dim in embedding_dims:
            print(f"Training with window size {window_size} and embedding dimension {embedding_dim}")
            training_pairs = build_training_pairs(words, window_size)
            if not training_pairs:
                print(f"No training pairs generated for window size {window_size}. Skipping.")
                continue

            # Initialize model
            model = SkipGramModel(vocab_size=len(vocab), embedding_dim=embedding_dim)

            # Training loop
            epochs = 100
            for epoch in range(epochs):
                total_loss = 0
                for target_word, context_word in training_pairs:
                    if target_word not in word2idx or context_word not in word2idx:
                        continue
                    # Convert words to one-hot vector
                    target_vector = np.zeros(len(vocab))
                    target_vector[word2idx[target_word]] = 1

                    context_vector = np.zeros(len(vocab))
                    context_vector[word2idx[context_word]] = 1

                    # forward pass
                    hidden, output = model.forward(target_vector)

                    # compute loss
                    loss = -np.log(output[word2idx[context_word]])

                    # backward pass
                    model.backward(target_vector, context_vector)

                    total_loss += loss

                print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(training_pairs)}")

            # Output embeddings
            print(f"Embeddings for window size {window_size} and embedding dimension {embedding_dim}")
            for word in word2idx:
                word_idx = word2idx[word]
                word_vector = model.W1[word_idx]
                print(f"{word}: {word_vector}")

            # Evaluasi kedekatan kata
            test_words = list(word2idx.keys())[:5]  # Cek beberapa kata pertama
            for test_word in test_words:
                similar_words = model.get_most_similar(test_word, word2idx)
                print(f"Most similar words to '{test_word}': {similar_words}")


if __name__ == "__main__":
    api_key = "44dfd24144424d83bebfd58813ab6cf7"  # Ganti dengan API key yang valid
    train_example(api_key)
