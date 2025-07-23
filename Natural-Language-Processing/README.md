# Natural Language Processing (NLP) with PyTorch

This project demonstrates natural language processing techniques using both traditional machine learning approaches and deep learning with PyTorch. The example shows how to train text classifiers on word frequency counts using bag-of-words models and neural networks with word embeddings.

## PyTorch

The PyTorch version replaces TensorFlow/Keras components with PyTorch equivalents.

```python
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
```

## Original TensorFlow/Keras

The original implementation uses TensorFlow/Keras.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns
import string
import pickle

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
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
```

## Embeddings

**Word embeddings** are dense vector representations of words that capture semantic meaning and relationships. Unlike traditional one-hot encoding where each word is represented as a sparse vector with only one non-zero element, embeddings map words to dense vectors in a continuous vector space where semantically similar words are positioned closer together.

### Word Embeddings
- **Dense representation**: Each word is represented as a vector of real numbers (typically 100-300 dimensions)
- **Semantic similarity**: Words with similar meanings have similar vector representations
- **Contextual relationships**: Mathematical operations on embeddings can reveal relationships (e.g., king - man + woman ≈ queen)
- **Learned from data**: Embeddings are typically learned from large text corpora using algorithms like Word2Vec, GloVe, or FastText

## Bag-of-Words

**Bag-of-Words (BoW)** is an approach where:
- Text is represented as a collection of words, disregarding grammar and word order
- Each document is represented as a vector where each dimension corresponds to a word in the vocabulary
- Values can be binary (word present/absent) or frequency-based (word count or TF-IDF)
- Results in sparse, high-dimensional vectors
- Loses semantic information and word relationships

**Word embeddings** overcome BoW limitations by:
- Capturing semantic meaning and word relationships
- Providing dense, lower-dimensional representations
- Enabling transfer learning from pre-trained models
- Better handling of out-of-vocabulary words (with subword embeddings)

**Bag-of-Words**

The example shows how to do text classification starting from a text document.

**Newsgroups**

```
categories = [
    'comp.graphics',
    'rec.motorcycles',
    'sci.electronics',
    'talk.religion.misc'
]
```
**Load train and test data from fetch_20newsgroups**

```

rng = np.random.RandomState(42)

train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=rng)

test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=rng)
```
**Print out keys and data for the first file in the train set**
```
train.keys()

print(f"Data: {train.data[0][:40]}\n")
print(f"Filename: {train.filenames[0]}\n")
print(f"Category: {train.target_names[0]}\n")
print(f"Category, numeric label: {train.target[0]}")
```

**Create a counter object from targets (category) of train and test sets**

```
train_counter = Counter(train.target)
test_counter =  Counter(test.target)
```

**Create dataframe with counted files belonging to a certain category**

```
cl = pd.DataFrame(data={
    'Train': { **{ train.target_names[index]: count for index, count in train_counter.items()}, 'Total': len(train.target)},
    'Test':  { **{test.target_names[index]: count for index, count in test_counter.items()},  'Total': len(test.target)},
})

cl.columns = pd.MultiIndex.from_product([["Class distribution"], cl.columns])
cl
```

![alt text](https://github.com/jylhakos/Deep-Learning-and-PyTorch/blob/main/Natural-Language-Processing/1.png?raw=true)

**Transform text to a TF-IDF-weighted document-term matrix**

The vocabulary is a list of words that occurred in the text document where each word has its own index. 

Tokenizer generates the dictionary of word encodings and creates vectors from sentences. 

```
# Text document i.e. a collection of linguistic data
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

```

**A pre-processing pipeline to convert documents to vectors using bag-of-words and TF-IDF**


``` 
# Remove numbers, lines like - or _ or combinations like "/|", "||/" or "////"

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[-_]+', '', text)
    text = re.sub(r'\/*\|+|\/+', '', text)
    return text

```

**The collection of linguistic data pre-processing pipeline**

The inputs to the function are:
- list of strings (one element is a document)
- maximum number of features (size of the vocabulary) to use

Returns a Pipeline object

```
def text_processing_pipeline(features=None):
	vectorizer = CountVectorizer(preprocessor=preprocess_text, analyzer='word', stop_words='english', max_features=features)
    tfidf = TfidfTransformer(smooth_idf=True,use_idf=True)
    pipeline = Pipeline([('count', vectorizer),('tfid', tfidf)]).fit(corpus)
    return pipeline

```

**CountVectorizer**

Scikit-learn’s CountVectorizer is used to convert a collection of text documents to a vector of term or token counts. 

TfidfTransformer.

Transform a count matrix to a normalized tf or tf-idf representation.

```
tfidf = TfidfTransformer(smooth_idf=True,use_idf=True)

```

**Using text processing pipeline to create training and test datasets**

```
pipeline = text_processing_pipeline(features=10000)
X_train = pipeline.fit_transform(train.data)
y_train = train.target

X_test = pipeline.transform(test.data)
y_test = test.target
```

**multi-class classification**

```
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
```

**The confusion matrix**

![alt text](https://github.com/jylhakos/Deep-Learning-and-PyTorch/blob/main/Natural-Language-Processing/2.png?raw=true)

**Document classification with word Embeddings**

One way to represent the text is to convert sentences into embeddings vectors.

```
import pickle

from pathlib import Path

embeddings_path = Path().cwd() / '..' / '..' / '..' / 'coursedata' / 'R5' / '20newsgroups_subset_vocabulary_embeddings.p'

with open(embeddings_path, "rb") as f:
    embeddings = pickle.load(f)
    vocabulary = list(embeddings.keys())
    
print(f'The vocabulary has a total of {len(vocabulary)} words')

# Extract a training & validation split
validation_split = 0.2
num_validation_samples = int(validation_split * len(train.data))

train_samples = train.data[:-num_validation_samples]
val_samples = train.data[-num_validation_samples:]

train_labels = train.target[:-num_validation_samples]
val_labels = train.target[-num_validation_samples:]

test_samples = test.data
test_labels = test.target

```
**Using PyTorch CNN to perform document classification**

PyTorch implementation replaces Keras Sequential model with a custom nn.Module:

```python
class TextCNN(nn.Module):
    def __init__(self, num_tokens, embedding_dim, embedding_matrix, num_classes, freeze_embeddings=True):
        super(TextCNN, self).__init__()
        
        # Embedding layer (replaces Keras Embedding)
        self.embedding = nn.Embedding(num_tokens, embedding_dim)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False
            
        # Convolutional layers (replaces Keras Conv1D)
        self.conv1 = nn.Conv1d(embedding_dim, 128, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=5, padding=2) 
        self.conv3 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        
        # Pooling layers (replaces Keras MaxPool1D and GlobalMaxPool1D)
        self.maxpool1 = nn.MaxPool1d(2)
        self.maxpool2 = nn.MaxPool1d(2)
        self.global_maxpool = nn.AdaptiveMaxPool1d(1)
        
        # Dense layers (replaces Keras Dense)
        self.dense = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)  # Conv1d expects (batch, channels, length)
        
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = F.relu(self.conv3(x))
        x = self.global_maxpool(x)
        
        x = x.squeeze(-1)
        x = F.relu(self.dense(x))
        x = self.dropout(x)
        x = self.output(x)
        return x

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TextCNN(num_tokens, embedding_dim, embedding_matrix, num_classes)
model.to(device)

# Loss function and optimizer (replaces model.compile)
criterion = nn.CrossEntropyLoss()  # replaces 'sparse_categorical_crossentropy'
optimizer = optim.RMSprop(model.parameters(), lr=0.001)  # replaces 'RMSprop'

# Training loop (replaces model.fit)
for epoch in range(num_epochs):
    model.train()
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

# Save PyTorch model
torch.save(model.state_dict(), 'model_pytorch.pth')
```

**Original Keras code**

```python
embedding_layer = Embedding(
    num_tokens,
    embedding_dim,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    trainable=False,
)

x_train = vectorizer(np.array([[s] for s in train_samples])).numpy()
x_val   = vectorizer(np.array([[s] for s in val_samples])).numpy()
x_test  = vectorizer(np.array([[s] for s in test_samples])).numpy()

y_train = train_labels
y_val   = val_labels
y_test  = test_labels

# The vocabulary size, or the number of words in the vocabulary built by Tokenizer

vocabulary_size = 20000

print(vocabulary_size)

# The vocabulary has a total of 20000 words

# Each document is of length 500 tokens.

# The embedding vectors of length 300.

print(f"Training set shape: {x_train.shape}")
print(f"Validation set shape: {x_val.shape}")
print(f"Test set shape: {x_test.shape}")

x_train_emb = embedding_layer(x_train)

# Number of categories for classification
m = len(categories)

print(m)

print(x_train_emb.shape)

model = keras.Sequential([
    embedding_layer,
    layers.Conv1D(filters=128, kernel_size=5, activation="relu", name="cv1"),
    layers.MaxPool1D(pool_size=2, name="maxpool1"),
    layers.Conv1D(filters=128, kernel_size=5, activation="relu", name="cv2"),
    layers.MaxPool1D(pool_size=2, name="maxpool2"),
    layers.Conv1D(filters=128, kernel_size=5, activation="relu", name="cv3"),
    layers.GlobalMaxPool1D(name="globalmaxpool"),
    layers.Dense(128, activation="relu", name="dense"),
    layers.Dropout(0.5),
    layers.Dense(m, activation='softmax', name="output")
])

model.summary()

# Compile and train the model
model.compile(optimizer='RMSprop', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

if training:
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=20, verbose=1)
    model.save('model.h5')
```

## Differences: TensorFlow/Keras vs PyTorch

### Model
- **Keras**: Uses `Sequential` API for simple model stacking
- **PyTorch**: Requires custom `nn.Module` class with explicit `forward()` method

### Training
- **Keras**: `model.compile()` + `model.fit()` handles training automatically
- **PyTorch**: Manual training loop with explicit gradient computation and optimization steps

### Flexibility
- **Keras**: Higher-level API, easier for beginners
- **PyTorch**: More explicit control, better for research and custom architectures

## Virtual Python Environment

Create and activate a Python virtual environment:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Linux/Mac)
source .venv/bin/activate

# Install required packages
pip install torch torchvision torchaudio numpy pandas matplotlib seaborn scikit-learn nltk jupyter ipykernel
```

## Running the code

1. **Python Script**: Run the PyTorch version with `python "Natural Language Processing-PyTorch.py"`
2. **Jupyter Notebook**: Open and run `Round 5 - Natural Language Processing (NLP)-PyTorch.ipynb`
3. **Original Files**: The original TensorFlow/Keras files are preserved as `Natural Language Processing.py` and `Round 5 - Natural Language Processing (NLP).ipynb`

## Dataset location

The dataset files are located in `../Dataset/R5/` relative to the project directory. Make sure the Dataset folder is available at the correct path.

![alt text](https://github.com/jylhakos/Deep-Learning-and-PyTorch/blob/main/Natural-Language-Processing/3.png?raw=true)

training=True

# Compile the model
model.compile(optimizer='RMSprop', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

# Training the model
if training:
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=20, verbose=1)
    model.save('model.h5')
else:
    model = tf.keras.models.load_model("model.h5")

model = tf.keras.models.load_model("model.h5")
_, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("Test accuracy {:.2f}".format(test_acc))
```

**An end-to-end NLP model**

```
# Use Keras layer that takes a string as an input
string_input = keras.Input(shape=(1,), dtype="string")

# Vectorize string input
x = vectorizer(string_input)

# Pass to main model
preds = model(x)

end_to_end_model = keras.Model(string_input, preds)
end_to_end_model.summary()

y_pred = [np.argmax(prob) for prob in end_to_end_model.predict(test.data)]

score = accuracy_score(y_test, y_pred)
print("Accuracy:   %0.3f" % score)

f1 = f1_score(y_test, y_pred, average='weighted')
print("      F1:   %0.3f" % f1)

from sklearn.metrics import multilabel_confusion_matrix
cfs_matrix = multilabel_confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    
for axes, cfs, label in zip(ax.flatten(), cfs_matrix, train.target_names):
    print_confusion_matrix(cfs, axes, label, ["N", "P"])

fig.tight_layout()
plt.show()
```

![alt text](https://github.com/jylhakos/Deep-Learning-and-PyTorch/blob/main/Natural-Language-Processing/4.png?raw=true)


**References**

PyTorch NLP Tutorials:
- https://pytorch.org/tutorials/beginner/deep_learning_nlp_tutorial.html
- https://pytorch.org/tutorials/intermediate/nlp_from_scratch_index.html

The 20 newsgroups text dataset:
- https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html

Original TensorFlow/Keras references:
- https://www.tensorflow.org/tutorials/text/word_embeddings
- https://www.tensorflow.org/tutorials/text/text_classification_rnn

## Project

```
Natural-Language-Processing/
├── .venv/                                          # Python virtual environment (PyTorch)
├── .gitignore                                      # Git ignore file (updated)
├── README.md                                       # This file (updated with PyTorch info)
├── Natural Language Processing.py                  # Original TensorFlow/Keras implementation (✅ paths fixed)
├── Natural Language Processing-PyTorch.py          # New PyTorch implementation (✅ working)
├── Round 5 - Natural Language Processing (NLP).ipynb          # Original Jupyter notebook (TensorFlow/Keras)
├── Round 5 - Natural Language Processing (NLP)-PyTorch.ipynb  # New PyTorch Jupyter notebook (✅ working)
├── test_embeddings_path.py                        # Test script to verify path checking
├── model_pytorch.pth                              # Saved PyTorch model
├── model_complete_pytorch.pth                     # Complete PyTorch model with metadata
└── *.png                                          # Images for README
```

## Steps

### For PyTorch Implementation (Recommended)

1. **Navigate to the project directory**
2. **The virtual environment is already set up with PyTorch**
3. **Run the code:**
   - **Python script**: `/home/laptop/EXERCISES/DEEP-LEARNING/PYTORCH/Deep-Learning-and-PyTorch/Natural-Language-Processing/.venv/bin/python "Natural Language Processing-PyTorch.py"`
   - **Jupyter notebook**: Open `Round 5 - Natural Language Processing (NLP)-PyTorch.ipynb`

### For TensorFlow/Keras Implementation (Original)

**Note**: The original TensorFlow/Keras files require TensorFlow to be installed:

1. **Install TensorFlow** (in a separate environment):
   ```bash
   pip install tensorflow scikit-learn nltk pandas matplotlib seaborn
   ```
2. **Run the original script**
   ```bash
   python "Natural Language Processing.py"
   ```

## Path verification

Both implementations now include robust path checking for the embeddings file. To test the path checking:

```bash
/home/laptop/EXERCISES/DEEP-LEARNING/PYTORCH/Deep-Learning-and-PyTorch/Natural-Language-Processing/.venv/bin/python test_embeddings_path.py
```

This will show you exactly which paths are being checked and whether the embeddings file is found.

## Dataset location

The dataset embeddings are expected at several possible locations:
1. `../Dataset/R5/20newsgroups_subset_vocabulary_embeddings.p`
2. `../../Dataset/R5/20newsgroups_subset_vocabulary_embeddings.p` 
3. `../../../Dataset/R5/20newsgroups_subset_vocabulary_embeddings.p`
4. `../../../coursedata/R5/20newsgroups_subset_vocabulary_embeddings.p`
5. `Dataset/R5/20newsgroups_subset_vocabulary_embeddings.p` (current directory)
6. `20newsgroups_subset_vocabulary_embeddings.p` (current directory)

**If the embeddings file is not found**
- The code will automatically use random embeddings instead
- The model will still work but may have lower accuracy
- You can create test embeddings using the provided code in the notebook

**To create your own embeddings file for testing**
```python
# Run this in the notebook to create simple test embeddings
def create_test_embeddings():
    vocab = vectorizer.get_vocabulary()[:1000]
    embeddings_dict = {}
    for word in vocab:
        embeddings_dict[word] = np.random.normal(0, 0.1, 300)
    
    with open('20newsgroups_subset_vocabulary_embeddings.p', 'wb') as f:
        pickle.dump(embeddings_dict, f)
```

## Comparison

- **Logistic Regression (Baseline)**: ~92.7% accuracy
- **PyTorch CNN**: ~66.1% accuracy (without pre-trained Embeddings)

**Conclusions**

The PyTorch CNN shows good learning progress and would likely achieve better results with pre-trained Embeddings or more training epochs.


