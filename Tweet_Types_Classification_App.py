import streamlit as st
import torch
import torch.nn as nn
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

class_names = ['Hate Speech', 'Offensive Language', 'Neither']

with open("tokenizer_vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

vocab_size = len(vocab)
embedding_dim = 100
hidden_dim = 256
output_dim = 3
num_layers = 2
dropout = 0.3
max_length = 30

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        last_hidden = hidden[-1]
        out = self.dropout(last_hidden)
        out = self.fc(out)
        return out

@st.cache_resource
def load_model():
    model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=num_layers, dropout=dropout)
    model.load_state_dict(torch.load("best_lstm_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def text_to_sequence(tokens, vocab):
    return [vocab.get(word, vocab['<UNK>']) for word in tokens]

def pad_sequence(seq, max_length):
    if len(seq) >= max_length:
        return seq[:max_length]
    else:
        return seq + [vocab['<PAD>']] * (max_length - len(seq))

st.title("Hate Speech Detector")
st.write("Enter text below to classify it as Hate Speech, Offensive Language, or Neither.")

user_input = st.text_area("Enter text...")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        tokens = preprocess_text(user_input)
        sequence = text_to_sequence(tokens, vocab)
        padded = pad_sequence(sequence, max_length)
        input_tensor = torch.tensor([padded], dtype=torch.long)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            predicted_idx = torch.argmax(probabilities).item()

        st.subheader(f"Prediction: {class_names[predicted_idx]}")
        st.write(f"Confidence: {probabilities[predicted_idx].item()*100:.2f}%")

        st.bar_chart({class_names[i]: probabilities[i].item() for i in range(len(class_names))})