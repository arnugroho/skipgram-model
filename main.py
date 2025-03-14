import numpy as np

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
        exp_x = np.exp(x-np.max(x))
        return exp_x / exp_x.sum()

def train_example():
  vocab = ["I","like","deep","learning"]
  word2idx = {word: idx for idx, word in enumerate(vocab)}

  training_pairs = [
      ("I", "like"),
      ("like", "I"),
      ("like", "deep"),
      ("deep", "like"),
      ("deep", "learning"),
      ("learning", "deep")
  ]

  # Initialize model
  model = SkipGramModel(vocab_size=len(vocab), embedding_dim=20)

  # Training loop
  epochs = 100
  for epoch in range(epochs):
    total_loss = 0
    for target_word, context_word in training_pairs:
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

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(training_pairs)}")

  return model, word2idx

if __name__ == "__main__":
  model, word2idx = train_example()

  # get word embeddings
  for word in word2idx:
    word_idx = word2idx[word]
    word_vector = model.W1[word_idx]
    print(f"{word}: {word_vector}")

