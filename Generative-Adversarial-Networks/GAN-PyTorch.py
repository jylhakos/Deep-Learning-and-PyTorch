import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt 
import IPython.display as ipd
from IPython.display import IFrame

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

training = True

def sample_true_distr(n=10000):
    """
    Function to generate samples from the quadratic distribution.
    Takes as input sample size `n` and returns set of n-pairs (x,y),
    where x is a real number and y is a square of x plus a constant.
    """
    
    # set random seed for reproducibility
    np.random.seed(42)
    # draw samples from normal distribution
    x = 10*(np.random.random_sample((n,))-0.5)
    # compute y
    y = 10 + x*x

    return np.array([x, y]).T

# Create a dataset with samples from the quadratic distribution
target_data = sample_true_distr()

# Plot the first 32 datapoints
plt.scatter(target_data[:32,0],target_data[:32,1])
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.show()

batch_size = 32

# Convert numpy array to PyTorch tensor and create DataLoader
target_tensor = torch.tensor(target_data, dtype=torch.float32)
dataset = TensorDataset(target_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

print(f"Dataset created with {len(dataset)} samples")

# size of the random vector
codings_size = 10

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

# Initialize models
generator = Generator(codings_size).to(device)
discriminator = Discriminator().to(device)

print("Generator architecture:")
print(generator)
print("\nDiscriminator architecture:")
print(discriminator)

# Initialize optimizers
lr = 0.0002
generator_optimizer = optim.RMSprop(generator.parameters(), lr=lr)
discriminator_optimizer = optim.RMSprop(discriminator.parameters(), lr=lr)

# Loss function
criterion = nn.BCELoss()

def train_gan(generator, discriminator, dataloader, batch_size, codings_size, n_epochs=90):
    saved_samples = np.zeros((int(n_epochs/10), 2, batch_size, 2))
    
    generator.train()
    discriminator.train()
    
    for epoch in range(n_epochs):
        for i, (X_batch,) in enumerate(dataloader):
            X_batch = X_batch.to(device)
            
            # Phase 1 - training the Discriminator
            discriminator_optimizer.zero_grad()
            
            # Generate fake samples
            noise = torch.randn(batch_size, codings_size, device=device)
            gen_samples = generator(noise)
            
            # Concatenate real and fake samples
            X_fake_and_real = torch.cat([gen_samples.detach(), X_batch], dim=0)
            
            # Create labels (0 for fake, 1 for real)
            labels = torch.cat([
                torch.zeros(batch_size, 1, device=device),
                torch.ones(batch_size, 1, device=device)
            ], dim=0)
            
            # Forward pass through discriminator
            disc_output = discriminator(X_fake_and_real)
            disc_loss = criterion(disc_output, labels)
            
            # Backward pass and optimize discriminator
            disc_loss.backward()
            discriminator_optimizer.step()
            
            # Phase 2 - training the Generator
            generator_optimizer.zero_grad()
            
            # Generate new noise and fake samples
            noise = torch.randn(batch_size, codings_size, device=device)
            gen_samples = generator(noise)
            
            # Labels for generator (we want discriminator to think they're real)
            gen_labels = torch.ones(batch_size, 1, device=device)
            
            # Forward pass through discriminator (for generator training)
            disc_output_gen = discriminator(gen_samples)
            gen_loss = criterion(disc_output_gen, gen_labels)
            
            # Backward pass and optimize generator
            gen_loss.backward()
            generator_optimizer.step()
            
            break  # Only process first batch per epoch (matching original behavior)
            
        if epoch % 10 == 0:
            print("Epoch {}/{}".format(epoch, n_epochs))
            with torch.no_grad():
                # Save samples for plotting
                noise = torch.randn(batch_size, codings_size, device=device)
                gen_samples = generator(noise)
                saved_samples[int(epoch/10), 0, :, :] = X_batch.cpu().numpy()
                saved_samples[int(epoch/10), 1, :, :] = gen_samples.cpu().numpy()

    return saved_samples

if training:
    saved_samples = train_gan(generator, discriminator, dataloader, batch_size, codings_size)

if training:
    plt.figure(figsize=(8, 8))

    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        # plot real samples
        plt.scatter(saved_samples[i,0,:,0], saved_samples[i,0,:,1], label='Real', alpha=0.7)
        # plot generated (fake) samples
        plt.scatter(saved_samples[i,1,:,0], saved_samples[i,1,:,1], label='Generated', alpha=0.7)
        plt.legend()
        plt.axis("off")
    plt.show()

# Save models
if training:
    torch.save(generator.state_dict(), 'generator_pytorch.pth')
    torch.save(discriminator.state_dict(), 'discriminator_pytorch.pth')
    print("Models saved successfully!")
