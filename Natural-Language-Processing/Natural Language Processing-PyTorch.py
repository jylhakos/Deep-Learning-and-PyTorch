# Natural Language Processing with PyTorch
# Original TensorFlow/Keras implementation converted to PyTorch
# https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html
# https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
# https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns
import string
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence

from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import multilabel_confusion_matrix
from pathlib import Path
from nltk import word_tokenize
from nltk.corpus import stopwords

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Bag-of-Words

rng = np.random.RandomState(42)

# Newsgroups
categories = [
    'comp.graphics',
    'rec.motorcycles',
    'sci.electronics',
    'talk.religion.misc'
]

# Load train and test data from fetch_20newsgroups
train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=rng)
test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=rng)

train.keys()

print(f"Data: {train.data[0][:40]}\n")
print(f"Filename: {train.filenames[0]}\n")
print(f"Category: {train.target_names[0]}\n")
print(f"Category, numeric label: {train.target[0]}")


# Summary of train and test datasets

# Create a counter object from targets (category) of train and test sets
train_counter = Counter(train.target)
test_counter =  Counter(test.target)

# Create dataframe with counted n.o. files belonging to a certain category
cl = pd.DataFrame(data={
    'Train': { **{ train.target_names[index]: count for index, count in train_counter.items()}, 'Total': len(train.target)},
    'Test':  { **{test.target_names[index]: count for index, count in test_counter.items()},  'Total': len(test.target)},
})

cl.columns = pd.MultiIndex.from_product([["Class distribution"], cl.columns])
cl

# Text document
corpus = ['this is the first document',
          'this document is the second document',
          'and this is the third one',
          'is this the first document']

# Vocabulary
vocabulary = ['this', 'document', 'first', 'is', 'second', 'the',
              'and', 'one']

# Create tokens from text given vocabulary (i.e. create the count vectorizer)
token_matrix = CountVectorizer(vocabulary=vocabulary)

# Convert count matrix to TF-IDF format (i.e.create the tfi-df trasformer)
tfid_transform = TfidfTransformer()

# Chain steps
pipe = Pipeline([('count', token_matrix),
                 ('tfid', tfid_transform)])

# Fit data
pipe.fit(corpus)

# Display tokenized text
pipe['count'].transform(corpus).toarray()

# Display text converted to TF-IDF representation
pipe.transform(corpus).toarray()

# A pre-processing pipeline to convert documents to vectors using bag-of-words and TF-IDF

# Remove numbers, lines like - or _ or combinations like "/|", "||/" or "////"

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[-_]+', '', text)
    text = re.sub(r'\/*\|+|\/+', '', text)
    return text


'''
Corpus pre-processing pipeline

The inputs to the function are:
- list of strings (one element is a document)
- maximum number of features (size of the vocabulary) to use  
Returns a Pipeline object   
'''
def text_processing_pipeline(features=None):
    vectorizer = CountVectorizer(preprocessor=preprocess_text, analyzer='word', stop_words='english', max_features=features)
    tfidf = TfidfTransformer(smooth_idf=True,use_idf=True)
    pipeline = Pipeline([('count', vectorizer),('tfid', tfidf)]).fit(corpus)
    return pipeline


pipeline = text_processing_pipeline(features=10000)

X_train = pipeline.fit_transform(train.data)
y_train = train.target

X_test = pipeline.transform(test.data)
y_test = test.target

# define classifier with sklearn LogisticRegression
clf = LogisticRegression(random_state=0)

# fit classifier to training set
clf = clf.fit(X_train, y_train)

# get predictions for test set
pred = clf.predict(X_test)

score = accuracy_score(y_test, pred)
print("Accuracy:   %0.3f" % score)

f1 = f1_score(y_test, pred, average='weighted')
print("      F1:   %0.3f" % f1)

def print_confusion_matrix(confusion_matrix, axes, class_label, class_names, fontsize=14):

    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    axes.set_ylabel('True label')
    axes.set_xlabel('Predicted label')
    axes.set_title("Confusion Matrix for the class - " + class_label)

cfs_matrix = multilabel_confusion_matrix(y_test, pred)

fig, ax = plt.subplots(2, 2, figsize=(12, 7))

for axes, cfs, label in zip(ax.flatten(), cfs_matrix, train.target_names):
    print_confusion_matrix(cfs, axes, label, ["N", "P"])

fig.tight_layout()

plt.show()

# Document classification with word Embeddings using PyTorch

# Try different possible paths for the embeddings file
possible_paths = [
    Path().cwd() / '..' / 'Dataset' / 'R5' / '20newsgroups_subset_vocabulary_embeddings.p',
    Path().cwd() / '..' / '..' / 'Dataset' / 'R5' / '20newsgroups_subset_vocabulary_embeddings.p',
    Path().cwd() / '..' / '..' / '..' / 'Dataset' / 'R5' / '20newsgroups_subset_vocabulary_embeddings.p',
    Path().cwd() / '..' / '..' / '..' / 'coursedata' / 'R5' / '20newsgroups_subset_vocabulary_embeddings.p',
    Path().cwd() / 'Dataset' / 'R5' / '20newsgroups_subset_vocabulary_embeddings.p',
    Path().cwd() / '20newsgroups_subset_vocabulary_embeddings.p'
]

embeddings = None
embeddings_path = None

print("Searching for embeddings file in possible locations:")
for i, path in enumerate(possible_paths, 1):
    print(f"{i}. {path}")
    if path.exists():
        print(f"   ✓ Found!")
        embeddings_path = path
        break
    else:
        print(f"   ✗ Not found")

if embeddings_path and embeddings_path.exists():
    try:
        with open(embeddings_path, "rb") as f:
            embeddings = pickle.load(f)
            vocabulary = list(embeddings.keys())
        print(f'\n✓ Successfully loaded embeddings for {len(vocabulary)} words')
    except Exception as e:
        print(f"\n❌ Error loading embeddings file: {e}")
        print("Using random embeddings instead.")
        embeddings = None
else:
    print("\n⚠️  No embeddings file found. Using random embeddings instead.")
    embeddings = None

# Extract a training & validation split
validation_split = 0.2
num_validation_samples = int(validation_split * len(train.data))

train_samples = train.data[:-num_validation_samples]
val_samples = train.data[-num_validation_samples:]

train_labels = train.target[:-num_validation_samples]
val_labels = train.target[-num_validation_samples:]

test_samples = test.data
test_labels = test.target

# PyTorch Text Vectorizer (replacement for TensorFlow TextVectorization)
class TextVectorizer:
    def __init__(self, max_features=10000, max_len=100):
        self.max_features = max_features
        self.max_len = max_len
        self.word_index = {}
        self.vocabulary = []
        
    def fit(self, texts):
        # Build vocabulary
        word_counts = Counter()
        for text in texts:
            words = self._tokenize(text)
            word_counts.update(words)
        
        # Keep only most frequent words
        most_common = word_counts.most_common(self.max_features - 2)  # -2 for PAD and UNK
        
        # Create word to index mapping
        self.word_index = {'<PAD>': 0, '<UNK>': 1}
        self.vocabulary = ['<PAD>', '<UNK>']
        
        for word, _ in most_common:
            self.word_index[word] = len(self.vocabulary)
            self.vocabulary.append(word)
    
    def _tokenize(self, text):
        # Simple tokenization - split by spaces and remove punctuation
        text = preprocess_text(text)
        return text.split()
    
    def transform(self, texts):
        sequences = []
        for text in texts:
            words = self._tokenize(text)
            sequence = [self.word_index.get(word, 1) for word in words]  # 1 is UNK
            # Truncate or pad to max_len
            if len(sequence) > self.max_len:
                sequence = sequence[:self.max_len]
            else:
                sequence = sequence + [0] * (self.max_len - len(sequence))  # 0 is PAD
            sequences.append(sequence)
        return np.array(sequences)
    
    def get_vocabulary(self):
        return self.vocabulary

# Create PyTorch vectorizer
vectorizer = TextVectorizer(max_features=10000, max_len=100)
vectorizer.fit(train_samples + val_samples + test_samples)

sentence = "Robert Plant wrote a hell of a song"
output = vectorizer.transform([sentence])
print(f"Vectorized sentence shape: {output.shape}")
print(f"First 8 tokens: {output[0][:8]}")

voc = vectorizer.get_vocabulary()
word_index = {word: i for i, word in enumerate(voc)}

num_tokens = len(voc)
embedding_dim = 300
hits = 0
misses = 0

# Create embedding matrix for PyTorch
embedding_matrix = np.zeros((num_tokens, embedding_dim))

for word, i in word_index.items():
    embedding_vector = embeddings.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1

print(f"Converted {hits} words ({misses} misses)")

# PyTorch CNN Model for document classification
class TextCNN(nn.Module):
    def __init__(self, num_tokens, embedding_dim, embedding_matrix, num_classes, freeze_embeddings=True):
        super(TextCNN, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(num_tokens, embedding_dim)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False
            
        # Convolutional layers
        self.conv1 = nn.Conv1d(embedding_dim, 128, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        
        # Pooling layers
        self.maxpool1 = nn.MaxPool1d(2)
        self.maxpool2 = nn.MaxPool1d(2)
        self.global_maxpool = nn.AdaptiveMaxPool1d(1)
        
        # Dense layers
        self.dense = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        x = x.transpose(1, 2)  # (batch_size, embedding_dim, seq_len) for Conv1d
        
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = F.relu(self.conv3(x))
        x = self.global_maxpool(x)  # (batch_size, 128, 1)
        
        x = x.squeeze(-1)  # (batch_size, 128)
        x = F.relu(self.dense(x))
        x = self.dropout(x)
        x = self.output(x)
        
        return x

# Prepare data for PyTorch
x_train = vectorizer.transform(train_samples)
x_val = vectorizer.transform(val_samples)
x_test = vectorizer.transform(test_samples)

y_train = torch.tensor(train_labels, dtype=torch.long)
y_val = torch.tensor(val_labels, dtype=torch.long)
y_test = torch.tensor(test_labels, dtype=torch.long)

print(f"Training set shape: {x_train.shape}")
print(f"Validation set shape: {x_val.shape}")
print(f"Test set shape: {x_test.shape}")

# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.long)
x_val_tensor = torch.tensor(x_val, dtype=torch.long)
x_test_tensor = torch.tensor(x_test, dtype=torch.long)

# Create datasets and dataloaders
train_dataset = TensorDataset(x_train_tensor, y_train)
val_dataset = TensorDataset(x_val_tensor, y_val)
test_dataset = TensorDataset(x_test_tensor, y_test)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Number of categories for classification
m = len(categories)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = TextCNN(num_tokens, embedding_dim, embedding_matrix, m)
model.to(device)

print(f"Model summary:")
print(model)

# Training settings
training = True

if training:
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 20
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        val_acc = 100 * correct / total
        val_accuracies.append(val_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    # Save model
    torch.save(model.state_dict(), 'model_pytorch.pth')
    print("Model saved as 'model_pytorch.pth'")
else:
    # Load model
    model.load_state_dict(torch.load('model_pytorch.pth'))
    print("Model loaded from 'model_pytorch.pth'")

# Evaluate on test set
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = model(batch_x)
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

test_acc = 100 * correct / total
print(f"Test accuracy: {test_acc:.2f}%")
