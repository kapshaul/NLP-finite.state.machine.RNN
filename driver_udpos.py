from collections import Counter
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtext import datasets
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from nltk.tokenize import word_tokenize

import matplotlib.pyplot as plt


FILE = 'bilstm_pos_model.pth'
# Parameter Setting
MODE = 'Train'          # {'Train', 'Test'}
VOCAB_SIZE = 15000      # Vocabulary size

num_epochs = 50         # Epoch number
batch_size = 64         # Batch size
learning_rate = 0.001   # Learning rate


######################################################################
# BI-LSTM model
class BILSTM_POS(nn.Module):
    def __init__(self, vocab_size, tag_size, embedding_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, max_norm=True)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, tag_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, s):
        x = self.embedding(x)
        x = pack_padded_sequence(x, s, batch_first=True, enforce_sorted=False)
        o, (h, c) = self.bilstm(x)
        o, _ = pad_packed_sequence(o, batch_first=True)
        out = self.dropout(o)
        out = self.fc(out)
        #out = torch.softmax(out, dim=-1)
        out = torch.log_softmax(out, dim=-1)
        return out

# Vocabulary
class Vocabulary:
    def __init__(self, corpus_tokens, tag_tokens):
        self.word2idx, self.idx2word, self.freq = self.build_vocab(corpus_tokens)
        self.tag2idx, self.idx2tag = self.build_POStag(tag_tokens)
        self.word2tag, self.tag2word = self.build_vocab_tag(corpus_tokens, tag_tokens)

        self.size = len(self.word2idx)

    def text2idx(self, tokens):
        return [self.word2idx[t.lower()] if t in self.word2idx.keys() else self.word2idx['UNK'] for t in tokens]

    def idx2text(self, idxs):
        return [self.idx2word[i] if i in self.idx2word.keys() else 'UNK' for i in idxs]

    def text2tag(self, tokens):
        return [self.word2tag[t] if t in self.word2tag.keys() else self.idx2tag[0] for t in tokens]

    def tokenize(self, text):
        # Convert to lowercase
        text = text.lower()
        # Tokenize the text
        tokens = word_tokenize(text)
        return tokens

    def build_vocab(self, corpus_tokens):
        """
        build_vocab takes in list of tokenizes, and builds a finite vocabulary

        :params:
        - corpus: a list string to build a vocabulary over

        :returns:
        - word2idx: a dictionary mapping token strings to their numerical index in the dictionary e.g. { "dog": 0, "but":1, ..., "UNK":129}
        - idx2word: the inverse of word2idx mapping an index in the vocabulary to its word e.g. {0: "dog", 1:"but", ..., 129:"UNK"}
        - freq: a dictionary of words and frequency counts over the corpus (including words not in the dictionary), e.g. {"dog":102, "the": 18023, ...}
        """
        # Flattening all the tokens in the training corpus
        flattened_text = [string.lower() for sublist in corpus_tokens for string in sublist]
        word_count_dict_counter = Counter(flattened_text)
        word_count_dict = word_count_dict_counter.most_common(VOCAB_SIZE)
        # Build word2idx, idx2word, and freq dictionaries
        word2idx, idx2word, freq = {}, {}, {}
        index = 0
        for word, frequency in word_count_dict:
            word2idx[word] = index
            idx2word[index] = word
            freq[word] = frequency
            index += 1
        word2idx['UNK'] = index

        return word2idx, idx2word, freq

    def build_POStag(self, tag_tokens):
        # Flattening all the tokens in the training corpus
        flattened_tag = [string for sublist in tag_tokens for string in sublist]
        tag_tokens_count_dict_counter = Counter(flattened_tag)
        tag_count_dict = tag_tokens_count_dict_counter.most_common()

        tag2idx, idx2tag = {}, {}
        index = 0
        for tag, frequency in tag_count_dict:
            tag2idx[tag] = index
            idx2tag[index] = tag
            index += 1

        return tag2idx, idx2tag

    def build_vocab_tag(self, corpus_tokens, tag_tokens):
        # Flattening all the tokens in the training corpus
        flattened_text = [string.lower() for sublist in corpus_tokens for string in sublist]
        # Flattening all the tokens in the training corpus
        flattened_tag = [string for sublist in tag_tokens for string in sublist]

        word2tag, tag2word = {}, {}
        for word, tag in zip(flattened_text, flattened_tag):
            word2tag[word] = tag
            tag2word[tag] = word

        return word2tag, tag2word

# Language Model
class LanguageModel:
    def __init__(self, vocab, model, lr=0.001):
        self.vocab = vocab
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

        self.train_losses, self.valid_losses = [], []

    def batch_process(self, xx, yy):
        xx_idx = [self.vocab.text2idx(batch) for batch in xx]
        xx_tensors = [torch.tensor(seq, dtype=torch.int32) for seq in xx_idx]
        xx_padded = pad_sequence(xx_tensors, batch_first=True, padding_value=self.vocab.word2idx['UNK'])

        yy_idx = [[self.vocab.tag2idx.get(tag, 0) for tag in batch] for batch in yy]
        yy_tensors = [torch.tensor(seq, dtype=torch.long) for seq in yy_idx]
        yy_padded = pad_sequence(yy_tensors, batch_first=True, padding_value=0)

        return xx_padded, yy_padded, yy_idx

    # Train and validate the model
    def train_model(self, train_loader, valid_loader, num_epochs=10):
        self.train_losses, self.valid_losses = [], []
        min_validation_loss = float('inf')

        for epoch in range(num_epochs):
            # Train model
            self.model.train()
            epoch_train_loss = 0
            train_count = 0
            for xx, yy, x_lens in train_loader:
                xx_padded, yy_padded, _ = self.batch_process(xx, yy)
                self.optimizer.zero_grad()
                outputs = self.model(xx_padded, x_lens)
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), yy_padded.view(-1))
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()
                train_count += 1

            # Validate model
            self.model.eval()
            epoch_valid_loss = 0
            valid_count = 0
            with torch.no_grad():
                for xx, yy, x_lens in valid_loader:
                    xx_padded, yy_padded, _ = self.batch_process(xx, yy)
                    outputs = self.model(xx_padded, x_lens)
                    loss = self.criterion(outputs.view(-1, outputs.size(-1)), yy_padded.view(-1))
                    epoch_valid_loss += loss.item()
                    valid_count += 1
            if min_validation_loss > (epoch_valid_loss / valid_count):
                min_validation_loss = epoch_valid_loss / valid_count
                torch.save(self.model.state_dict(), FILE)

            # Record losses
            self.train_losses.append(epoch_train_loss / train_count)
            self.valid_losses.append(epoch_valid_loss / valid_count)

            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {self.train_losses[-1]}, Valid Loss: {self.valid_losses[-1]}")

    # Evaluation
    def test_model(self, test_loader):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xx, yy, x_lens in test_loader:
                xx_padded, yy_padded, yy_idx = self.batch_process(xx, yy)
                outputs = self.model(xx_padded, x_lens)
                est_padded = torch.argmax(outputs, dim=-1)

                # Counting correct estimation
                estimation = [sublist[:length] for sublist, length in zip(est_padded.numpy().tolist(), x_lens)]
                for i in range(len(yy_idx)):
                    for j in range(len(yy_idx[i])):
                        if yy_idx[i][j] == estimation[i][j]:
                            correct += 1
                total += np.sum(x_lens)
        print(f"Test Accuracy: {correct / total * 100:.2f}%\n")

    # Plot loss
    def plot_loss(self):
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.valid_losses, label="Valid Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()


######################################################################
# Helper Functions
def pad_collate(batch):
    xx = [b[0] for b in batch]
    yy = [b[1] for b in batch]
    x_lens = [len(x) for x in xx]

    return xx, yy, x_lens

# Data Load Function
def data_loader():
    train_data = datasets.UDPOS(split='train')
    valid_data = datasets.UDPOS(split='valid')
    test_data = datasets.UDPOS(split='test')

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=pad_collate)
    valid_loader = DataLoader(valid_data, batch_size=32, shuffle=True, drop_last=True, collate_fn=pad_collate)
    test_loader = DataLoader(test_data, batch_size=5, shuffle=True, drop_last=True, collate_fn=pad_collate)

    word_data, tag_data = [], []
    for xx, yy, x_lens in train_loader:
        word_data += xx
        tag_data += yy

    return train_loader, valid_loader, test_loader, word_data, tag_data

##################################
# TASK 3.3 #######################
##################################
def tag_sentence(sentence, model, vocab):
    # Tokenize the sentence
    tokens = [vocab.tokenize(t) for t in sentence]
    seq = [len(t) for t in tokens]
    # Convert tokens to index
    batch_idx = [vocab.text2idx(t) for t in tokens]
    batch_tensors = [torch.tensor(seq, dtype=torch.int32) for seq in batch_idx]
    batch_padded = pad_sequence(batch_tensors, batch_first=True, padding_value=vocab.word2idx['UNK'])
    # Estimation
    est_tags = model(batch_padded, seq)
    est_padded = torch.argmax(est_tags, dim=-1)
    estimation = [sublist[:length] for sublist, length in zip(est_padded.numpy().tolist(), seq)]

    print("The given sentence: ", sentence)
    print("Estimated POS Tags:")
    for idxs in estimation:
        text_list = []
        for idx in idxs:
            text_list.append(vocab.idx2tag[idx])
        print(text_list)
    print("Real POS Tags:")
    for text in tokens:
        tags = vocab.text2tag(text)
        print(tags)


######################################################################
def main():
    # Load dataset
    train_loader, valid_loader, test_loader, word_data, tag_data = data_loader()

    match MODE:
        case 'Train':
            # Pre-Processing: Building vocabulary based on training dataset
            vocab = Vocabulary(word_data, tag_data)
            with open('Vocabulary.pkl', 'wb') as f:
                pickle.dump(vocab, f)

            # Initialize BI-LSTM and language model
            model = BILSTM_POS(vocab_size=len(vocab.word2idx), tag_size=len(vocab.tag2idx))
            lm = LanguageModel(vocab, model, lr=learning_rate)
            # Train and validate, language model
            lm.train_model(train_loader, valid_loader, num_epochs=num_epochs)
            # Plot
            lm.plot_loss()

        case 'Test':
            # Load the class instance from the file
            with open('Vocabulary.pkl', 'rb') as f:
                vocab = pickle.load(f)

            # Initialize BI-LSTM and language model
            model = BILSTM_POS(vocab_size=len(vocab.word2idx), tag_size=len(vocab.tag2idx))
            # Load the model from the file
            model.load_state_dict(torch.load(FILE))
            lm = LanguageModel(vocab, model)
            # Test, language model
            lm.test_model(test_loader)

            ##################################
            # TASK 3.3 #######################
            ##################################
            sentence = ["The old man the boat.",
                        "The complex houses married and single soldiers and their families.",
                        "The man who hunts ducks out on weekends."]
            tag_sentence(sentence, model, vocab)


######################################################################
if __name__== "__main__":
    main()