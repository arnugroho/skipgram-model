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


def build_training_pairs(words, window_size=2):
    training_pairs = []
    for i, target_word in enumerate(words):
        for j in range(-window_size, window_size + 1):
            if j == 0 or i + j < 0 or i + j >= len(words):
                continue
            training_pairs.append((target_word, words[i + j]))
    return training_pairs


def train_example(api_key):
    news_titles = fetch_news(api_key)
    words = preprocess_texts(news_titles)
    if not words:
        print("No words extracted from news articles. Training aborted.")
        return None, None

    word_counts = Counter(words)
    vocab = list(word_counts.keys())
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    training_pairs = build_training_pairs(words)

    if not training_pairs:
        print("No training pairs generated. Training aborted.")
        return None, None

    # Initialize model
    model = SkipGramModel(vocab_size=len(vocab), embedding_dim=20)

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

        if len(training_pairs) > 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(training_pairs)}")

    return model, word2idx


if __name__ == "__main__":
    api_key = "44dfd24144424d83bebfd58813ab6cf7"  # Ganti dengan API key yang valid
    model, word2idx = train_example(api_key)

    if model and word2idx:
        # get word embeddings
        for word in word2idx:
            word_idx = word2idx[word]
            word_vector = model.W1[word_idx]
            print(f"{word}: {word_vector}")
