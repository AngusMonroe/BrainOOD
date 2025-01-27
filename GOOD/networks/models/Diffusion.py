import torch
import torch.nn as nn
import torch.nn.functional as F


# Define a high-pass filter kernel
def high_pass_filter_kernel():
    # Simple 3x3 high-pass filter kernel
    kernel = torch.tensor([[-1, -1, -1],
                           [-1,  8, -1],
                           [-1, -1, -1]], dtype=torch.float32)
    kernel = kernel.view(1, 1, 3, 3)  # Reshape to 4D tensor for convolution
    return kernel


def low_pass_filter_kernel():
    # Simple 3x3 low-pass filter kernel (averaging kernel)
    kernel = torch.tensor([[1/9, 1/9, 1/9],
                           [1/9, 1/9, 1/9],
                           [1/9, 1/9, 1/9]], dtype=torch.float32)
    kernel = kernel.view(1, 1, 3, 3)  # Reshape to 4D tensor for convolution
    return kernel


# Apply the high-pass filter to a 2D tensor
def apply_pass_filter(x):
    device = x.device
    # kernel = high_pass_filter_kernel().to(device)
    kernel = low_pass_filter_kernel().to(device)
    # Assuming x is a 4D tensor with shape (batch_size, channels, height, width)
    x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions if needed
    x_filtered = F.conv2d(x, kernel, padding=1)  # Convolve with padding to keep the size
    return x_filtered.squeeze()  # Remove added dimensions


# Define the forward diffusion process
def forward_diffusion(x_0, t, noise_std=0.05, high_pass=False):
    """Adds Gaussian noise to the tensor `x_0` over time steps `t`."""
    # x_0 = x_0.view(-1, 100, 100)
    dim = x_0.shape[1]
    bz = int(x_0.shape[0] / dim)
    if high_pass:
        noise = apply_pass_filter(x_0)
    else:
        device = x_0.device
        rand_feat = torch.randn(dim, dim).to(device)
        noise = ((rand_feat @ rand_feat.T) / dim * noise_std).repeat(bz, 1, 1).view(-1, dim)
    return x_0 + t * noise


# Define the reverse process (denoising model)
class DenoiseModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DenoiseModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        device = x.device

        # Use t as an additional feature or conditioning input if needed
        t = torch.rand(1).to(device)  # Random time step between 0 and 1
        x_noisy = forward_diffusion(x, t)

        # Forward pass through the denoising model
        x_reconstructed = self.model(x_noisy)

        # Compute loss
        loss = self.loss(x_reconstructed, x)

        return x_reconstructed, loss

    def loss(self, x_reconstructed, x):
        criterion = nn.MSELoss()
        return criterion(x_reconstructed, x)