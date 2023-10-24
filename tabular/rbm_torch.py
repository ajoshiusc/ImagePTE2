import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class GaussianBernoulliRBM(nn.Module):
    def __init__(self, num_visible, num_hidden, k=1):
        super(GaussianBernoulliRBM, self).__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.k = k

        # Parameters
        self.weights = nn.Parameter(torch.randn(num_visible, num_hidden) * 0.01)
        self.visible_bias = nn.Parameter(torch.zeros(num_visible))
        self.hidden_bias = nn.Parameter(torch.zeros(num_hidden))
        self.visible_std = nn.Parameter(torch.ones(num_visible))  # standard deviation for visible units

    def sample_from_prob(self, probs):
        return F.relu(torch.sign(probs - torch.rand(probs.size())))

    def forward(self, v):
        # Activation for the hidden units given visible units
        h_probs = torch.sigmoid(F.linear(v, self.weights, self.hidden_bias))
        h_sample = self.sample_from_prob(h_probs)
        return h_probs, h_sample

    def backward(self, h):
        # Activation for the visible units given hidden units
        v_mean = F.linear(h, self.weights.t(), self.visible_bias)
        v_sample = torch.normal(mean=v_mean, std=self.visible_std)
        return v_mean, v_sample

    def free_energy(self, v):
        vbias_term = v.mv(self.visible_bias)
        wx_b = F.linear(v, self.weights, self.hidden_bias)
        hidden_term = wx_b.exp().add(1).log().sum(1)
        return (-hidden_term - vbias_term).mean()

    def train_step(self, v, optimizer):
        v0 = v.clone()
        h_probs, h_sample = self.forward(v0)

        for _ in range(self.k):
            v_probs, v_sample = self.backward(h_sample)
            h_probs, h_sample = self.forward(v_sample)

        loss = self.free_energy(v0) - self.free_energy(v_sample)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def reconstruct(self, v):
        h_probs, _ = self.forward(v)
        v_reconstructed, _ = self.backward(h_probs)
        return v_reconstructed

# Generate a small synthetic dataset of size 100x5 with continuous values
data_size = 100
num_features = 5
synthetic_continuous_data = np.random.randn(data_size, num_features)

# Convert to tensor
data_tensor_continuous = torch.FloatTensor(synthetic_continuous_data)

# Display first 5 samples of the continuous data
synthetic_continuous_data[:5]
