# Generative Adversarial Networks (GANs)

A generative model is used in a way that takes a training set, consisting of samples drawn from a distribution pdata, and the model learns to represent an estimate of that distribution somehow. The result is a probability distribution pmodel.

## What are Generative Models?

Generative models are machine learning models that learn to generate new data samples similar to the training data. Unlike discriminative models that classify or predict labels, generative models learn the underlying data distribution to create synthetic data.

## Discriminators and Generators in GANs

### Generator
The **Generator** is a neural network that takes random noise as input and produces fake data samples that should resemble the real training data. The generator's goal is to fool the discriminator by creating increasingly realistic samples.

### Discriminator  
The **Discriminator** is a neural network that acts as a binary classifier, distinguishing between real data samples (from the training set) and fake samples (generated by the generator). The discriminator's goal is to correctly identify fake samples.

### Why use both?
The adversarial training between generator and discriminator creates a competitive learning environment where both networks improve through competition, leading to high-quality generated samples.

## PyTorch

This project includes both TensorFlow/Keras and PyTorch implementations for comparison.

- **TensorFlow/Keras files**: `GAN.py`, `utils.py`, original notebook cells
- **PyTorch files**: `GAN-PyTorch.py`, `utils-PyTorch.py`, updated notebook cells

### PyTorch advantages for GANs:
- Dynamic computation graphs for flexible experimentation
- Explicit control over training loops
- Better debugging capabilities
- Seamless GPU acceleration

**Discriminator training**

1. Fetch batch of real data.

2. Sample random vector, feed to the generator to get fake data.

3. Feed real and fake data to the discriminator for classification.

4. Compute loss and backpropagate the error/loss through discriminator.

5. Adjust discriminator's weights.

The discriminator is penalized for classifying fake data as "real".

### TensorFlow/Keras implementation:
```python
discriminator = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=16,activation=tf.nn.leaky_relu, input_shape=[2]),
    tf.keras.layers.Dense(units=16,activation=tf.nn.leaky_relu),
    tf.keras.layers.Dense(units=1,activation=tf.nn.sigmoid)
], name="Discriminator")

discriminator.build(input_shape=(None,codings_size))

discriminator.summary()
```

### PyTorch implementation:
```python
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 16),
            nn.LeakyReLU(0.01),
            nn.Linear(16, 16),
            nn.LeakyReLU(0.01),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

discriminator = Discriminator().to(device)
```

**Generator training**

1. Sample random vector, feed to the generator to get fake data.

2. Feed fake data to the discriminator for classification.

3. Compute loss and backpropagate the error/loss through generator. Adjust generator's weights.

The generator is penalized if the discriminator classified fake data as "fake".

### TensorFlow/Keras implementation:
```python
codings_size = 10

generator = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=16,activation=tf.nn.leaky_relu,input_shape=[codings_size]),
    tf.keras.layers.Dense(units=16,activation=tf.nn.leaky_relu),
    tf.keras.layers.Dense(units=2) 
], name="Generator")

generator.build(input_shape=(None,codings_size))

generator.summary()
```

### PyTorch implementation:
```python
class Generator(nn.Module):
    def __init__(self, codings_size=10):
        super(Generator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(codings_size, 16),
            nn.LeakyReLU(0.01),
            nn.Linear(16, 16),
            nn.LeakyReLU(0.01),
            nn.Linear(16, 2)
        )
    
    def forward(self, x):
        return self.network(x)

generator = Generator(codings_size).to(device)
```

**Combine both ANNs with a model of GAN**

### TensorFlow/Keras implementation:
```python
gan = tf.keras.models.Sequential([generator, discriminator])

tf.keras.utils.plot_model(
    gan,
    show_shapes=True, 
    show_layer_names=True
)

discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")

discriminator.trainable = False

gan.compile(loss="binary_crossentropy", optimizer="rmsprop")
```

### PyTorch implementation:
```python
# PyTorch manages models separately with explicit optimizers
generator_optimizer = optim.RMSprop(generator.parameters(), lr=0.0002)
discriminator_optimizer = optim.RMSprop(discriminator.parameters(), lr=0.0002)
criterion = nn.BCELoss()
```

## Training pipeline comparison

### TensorFlow vs PyTorch training differences:

#### TensorFlow/Keras:
- Uses `model.compile()` to set optimizer and loss
- `train_on_batch()` method handles forward/backward pass
- `model.trainable = False` to freeze layers
- More automated but less transparent

#### PyTorch:
- Manual optimizer setup and training loops
- Explicit `zero_grad()`, `backward()`, `step()` calls  
- More control over training process
- Better for research and experimentation

## Setup

### Script (Recommended)
```bash
# Make setup script executable and run it
chmod +x setup.sh
./setup.sh
```

### Manual

#### For TensorFlow/Keras (original):
```bash
pip install tensorflow numpy matplotlib jupyter
python GAN.py
```

#### For PyTorch (migrated):
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install PyTorch and dependencies
pip install -r requirements.txt

# Test installation
python test_pytorch.py

# Run PyTorch version
python GAN-PyTorch.py

# Or run Jupyter notebook
jupyter notebook Round6_GAN.ipynb
```

### Usage

#### Running GAN (2D data):
```bash
# TensorFlow version
python GAN.py

# PyTorch version  
python GAN-PyTorch.py
```

#### Running CNN GAN (MNIST):
```bash
# Open and run the Jupyter notebook
jupyter notebook Round6_GAN.ipynb
# Navigate to the CNN sections for MNIST experiments
```

```

![alt text](https://github.com/jylhakos/Deep-Learning-and-PyTorch/blob/main/Generative-Adversarial-Networks/1.png?raw=true)

**Training discriminator and generator**

Phase 1 for training discriminator consists of the following steps:

1. sample random vectors of shape (batch_size, codings_size) from a normal distribution

2. feed the random vectors to the generator, which will output generated samples

3. concatenate fake and real samples in one set

4. create labels for fake ( 𝑦 = 0 ) and real ( 𝑦 = 1 ) samples

5. use the train_on_batch(features, labels) method to train the discriminator. This method runs a single gradient update on a single batch of data. 

Phase 2 for training generator consists of the following steps:

1. sample random vectors of shape (batch_size, codings_size) from a normal distribution

2. create labels for fake samples  𝑦 = 1

3. use the train_on_batch(features, labels) method to train gan. By applying this method to the GAN model we will tune the weights of the generator. Parameters of discriminator are not affected, as previously we set discriminator.trainable = False before compiling gan.

```
training = True

def train_gan(gan, dataset, batch_size, codings_size, n_epochs=90):
    saved_samples = np.zeros((int(n_epochs/10),2,batch_size,2))
    generator, discriminator = gan.layers
    for epoch in range(n_epochs):
        for X_batch in dataset:
            
            # phase 1 - training the Discriminator
            noise = tf.random.normal(shape=[batch_size, codings_size])
            gen_samples = generator(noise)
            X_fake_and_real = tf.concat([gen_samples, X_batch], axis=0)
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            discriminator.train_on_batch(X_fake_and_real, y1)
            
            # phase 2 - training the Generator
            noise = tf.random.normal(shape=[batch_size, codings_size])
            y2 = tf.constant([[1.]] * batch_size)
            gan.train_on_batch(noise, y2)
            
        if epoch%10 == 0:
            print("Epoch {}/{}".format(epoch, n_epochs))
            saved_samples[int(epoch/10),0,:,:] = X_batch
            saved_samples[int(epoch/10),1,:,:] = gen_samples

    return saved_samples

if training:
    saved_samples = train_gan(gan, dataset, batch_size, codings_size)

if training:
    plt.figure(figsize=(8, 8))

    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        # plot real samples
        plt.scatter(saved_samples[i,0,:,0],saved_samples[i,0,:,1])
        # plot generated (fake) samples
        plt.scatter(saved_samples[i,1,:,0],saved_samples[i,1,:,1])
        plt.axis("off")
    plt.show()

```

![alt text](https://github.com/jylhakos/Deep-Learning-and-PyTorch/blob/main/Generative-Adversarial-Networks/2.png?raw=true)

**MNIST dataset**

```
training=True

if training:
    # Load training set
    (X_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    
    # Reshape and rescale
    X_train = X_train.reshape(-1, 28, 28, 1)/255 * 2. - 1.
    
    # Change data type
    X_train = tf.cast(X_train, tf.float32)


if training:
    batch_size = 32
    dataset = tf.data.Dataset.from_tensor_slices(X_train)
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)
```

**Generator**

```
codings_size = 100

cv_generator = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=7*7*128,activation=tf.nn.leaky_relu, input_shape=[codings_size]),
    tf.keras.layers.Reshape([7, 7, 128]),
    tf.keras.layers.BatchNormalization(momentum=0.99),
    tf.keras.layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(momentum=0.99),
    tf.keras.layers.Conv2DTranspose(1, kernel_size=5, strides=2, padding="same", activation="tanh")
], name="Generator")
```

**Discriminator**

```
cv_discriminator = tf.keras.models.Sequential([
    
    tf.keras.layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.2)),
    tf.keras.layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=1,activation=tf.nn.sigmoid)
], name="Discriminator")
```

**GAN model**

```
cv_gan = tf.keras.models.Sequential([cv_generator, cv_discriminator])

tf.keras.utils.plot_model(
    cv_gan,
    show_shapes=True,
    show_layer_names=True
)

cv_discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")

cv_discriminator.trainable = False

cv_gan.compile(loss="binary_crossentropy", optimizer="rmsprop")


def plot_multiple_images(images, n_cols=None):
    n_cols = n_cols or len(images)
    n_rows = (len(images) - 1) // n_cols + 1
    if images.shape[-1] == 1:
        images = np.squeeze(images, axis=-1)
    plt.figure(figsize=(n_cols, n_rows))
    for index, image in enumerate(images):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(image, cmap="binary")
        plt.axis("off")

test_random_vector = tf.random.normal(shape=[batch_size, codings_size])

def train_gan(gan, dataset, batch_size, codings_size, n_epochs=2):
    
    generator, discriminator = gan.layers
    itr=0
    for epoch in range(n_epochs):
        print("Epoch {}/{}".format(epoch + 1, n_epochs))
        for X_batch in dataset:
            # Phase 1 - training the discriminator
            noise = tf.random.normal(shape=[batch_size, codings_size])
            generated_images = generator(noise)
            X_fake_and_real = tf.concat([generated_images, X_batch], axis=0)
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            discriminator.trainable = True
            discriminator.train_on_batch(X_fake_and_real, y1)
            
            # Phase 2 - training the generator
            noise = tf.random.normal(shape=[batch_size, codings_size])
            y2 = tf.constant([[1.]] * batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y2)
            
            if (itr%100 == 0):
                gen_images = generator.predict(test_random_vector)
                plot_multiple_images(gen_images, 8)
                plt.show()
                         
            itr+=1

if training:
    train_gan(cv_gan, dataset, batch_size, codings_size)

```

### PyTorch CNN Implementation:

```python
class CNNGenerator(nn.Module):
    def __init__(self, codings_size=100):
        super(CNNGenerator, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(codings_size, 7*7*128),
            nn.LeakyReLU(0.01),
        )
        
        self.conv_layers = nn.Sequential(
            nn.BatchNorm2d(128, momentum=0.01),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.network(x)
        x = x.view(-1, 128, 7, 7)
        x = self.conv_layers(x)
        return x

class CNNDiscriminator(nn.Module):
    def __init__(self):
        super(CNNDiscriminator, self).__init__()
        
        self.network = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

# Initialize models
cv_generator = CNNGenerator(codings_size).to(device)
cv_discriminator = CNNDiscriminator().to(device)

# Optimizers
cv_generator_optimizer = optim.RMSprop(cv_generator.parameters(), lr=0.0002)
cv_discriminator_optimizer = optim.RMSprop(cv_discriminator.parameters(), lr=0.0002)
```

## Differences: TensorFlow vs PyTorch

| Aspect | TensorFlow/Keras | PyTorch |
|--------|------------------|---------|
| Model Definition | `Sequential([...])` | `nn.Module` classes |
| Training | `model.fit()`, `train_on_batch()` | Manual loops with `optimizer.step()` |
| Computation | Static/Eager execution | Dynamic computation graphs |
| Debugging | More challenging | Easier with standard Python debugging |
| Research | Good for production | Preferred for research |
| Learning Curve | Gentler for beginners | Steeper but more flexible |

## Dataset path configuration

The project uses the Dataset folder located at `../Dataset/` relative to the code directory. This allows both TensorFlow and PyTorch implementations to access the same data:

- MNIST dataset: Automatically downloaded to `../Dataset/MNIST/`
- Custom datasets: Place in `../Dataset/` subdirectories

## Project

```
Generative-Adversarial-Networks/
├── GAN.py                 # Original TensorFlow implementation
├── GAN-PyTorch.py         # PyTorch implementation
├── utils.py               # TensorFlow utilities
├── utils-PyTorch.py       # PyTorch utilities  
├── Round6_GAN.ipynb       # Jupyter notebook (supports both frameworks)
├── README.md              # This file
├── .venv/                 # Python virtual environment
├── .gitignore            # Git ignore file
└── *.png                 # Result images

```

![alt text](https://github.com/jylhakos/Deep-Learning-and-PyTorch/blob/main/Generative-Adversarial-Networks/3.png?raw=true)

![alt text](https://github.com/jylhakos/Deep-Learning-and-PyTorch/blob/main/Generative-Adversarial-Networks/4.png?raw=true)

![alt text](https://github.com/jylhakos/Deep-Learning-and-PyTorch/blob/main/Generative-Adversarial-Networks/5.png?raw=true)

![alt text](https://github.com/jylhakos/Deep-Learning-and-PyTorch/blob/main/Generative-Adversarial-Networks/6.png?raw=true)

![alt text](https://github.com/jylhakos/Deep-Learning-and-PyTorch/blob/main/Generative-Adversarial-Networks/7.png?raw=true)

![alt text](https://github.com/jylhakos/Deep-Learning-and-PyTorch/blob/main/Generative-Adversarial-Networks/8.png?raw=true)

![alt text](https://github.com/jylhakos/Deep-Learning-and-PyTorch/blob/main/Generative-Adversarial-Networks/9.png?raw=true)

