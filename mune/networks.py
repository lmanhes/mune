import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Normal
from collections import OrderedDict


class ProprioEncoder(nn.Module):
    """
    Encode proprioception sensors (in_features,) to vector (32,)
    """
    def __init__(self, in_features):
        super().__init__()
        self.fc = nn.Linear(in_features, 32)

    @property
    def output_size(self):
        return 32

    def forward(self, x):
        return self.fc(x)


class ProprioDecoder(nn.Module):
    """
    Decode a temporal multi-modal embedding vector into proprioception sensors values
    """
    def __init__(self, state_dim, rnn_hidden_dim, out_features):
        super().__init__()
        self.fc = nn.Linear(state_dim + rnn_hidden_dim, out_features)

    def forward(self, state, rnn_hidden):
        return self.fc(torch.cat([state, rnn_hidden], -1))


class VisionEncoder(nn.Module):
    """
    Encode image observation (3, 64, 64) to vector (1024,)
    """
    def __init__(self):
        super().__init__()
        self.conv_encoder = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 32, 4, stride=2)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(32, 64, 4, stride=2)),
            ('relu2', nn.ReLU()),
            ('conv3', nn.Conv2d(64, 128, 4, stride=2)),
            ('relu3', nn.ReLU()),
            ('conv4', nn.Conv2d(128, 256, 4, stride=2)),
            ('relu4', nn.ReLU())
        ]))

    @property
    def output_size(self):
        return self(torch.randn((1, 3, 64, 64))).size(-1)

    def forward(self, x):
        x = self.conv_encoder(x)
        return torch.flatten(x, start_dim=1)


class VisionDecoder(nn.Module):
    """
    Decode multi-modal embedding vector into image observation (3, 64, 64)
    """
    def __init__(self, state_dim, rnn_hidden_dim):
        super().__init__()

        self.fc_state_rnn = nn.Linear(state_dim + rnn_hidden_dim, 1024)

        self.convt_decoder = nn.Sequential(OrderedDict([
            ('convt1', nn.ConvTranspose2d(1024, 128, 5, stride=2)),
            ('relu1', nn.ReLU()),
            ('convt2', nn.ConvTranspose2d(128, 64, 5, stride=2)),
            ('relu2', nn.ReLU()),
            ('convt3', nn.ConvTranspose2d(64, 32, 6, stride=2)),
            ('relu3', nn.ReLU()),
            ('convt4', nn.ConvTranspose2d(32, 3, 6, stride=2))
        ]))

    def forward(self, state, rnn_hidden):
        x = self.fc_state_rnn(torch.cat([state, rnn_hidden], -1))
        x = x.view(x.size(0), 1024, 1, 1)
        return self.convt_decoder(x)


class FusionLayer(nn.Module):
    """
    Fused multiple vectors from different sensorial modalities into a unique vector (256,)
    """
    def __init__(self, in_features):
        super().__init__()
        self.fc_1 = nn.Linear(in_features, 512)
        self.fc_2 = nn.Linear(512, 256)

    @property
    def output_size(self):
        return 256

    def forward(self, *inputs):
        x = torch.cat(inputs, -1)
        x = self.fc_1(x)
        return self.fc_2(x)


class Rssm(nn.Module):
    """
    Recurrent state-space model

    Key components:
        - Posterior dynamics: h_t+1 = f(h_t, s_t, a_t)
        - Prior dynamics: p(s_t+1 | h_t+1)
        - Recurrent state model: q(s_t | h_t, o_t)
    """
    def __init__(self, state_dim, action_dim, rnn_hidden_dim, hidden_dim=200):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rnn_hidden_dim = rnn_hidden_dim

        self.fc_state_action = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc_rnn_hidden = nn.Linear(rnn_hidden_dim, hidden_dim)

        self.rnn = nn.GRUCell(hidden_dim, rnn_hidden_dim)

    def prior(self, state, action, rnn_hidden):
        """
        h_t+1 = f(h_t, s_t, a_t)
        Compute prior p(s_t+1 | h_t+1)
        """
        hidden = F.relu(self.fc_state_action(torch.cat([state, action], dim=1)))
        rnn_hidden = self.rnn(hidden, rnn_hidden)
        hidden = F.relu(self.fc_rnn_hidden(rnn_hidden))
        return hidden, rnn_hidden
        #mean = self.fc_state_mean_prior(hidden)
        #stddev = F.softplus(self.fc_state_stddev_prior(hidden)) + self._min_stddev
        #return Normal(mean, stddev), rnn_hidden

    def posterior(self, rnn_hidden, embedded_obs):
        """
        Compute posterior q(s_t | h_t, o_t)
        """
        hidden = F.relu(self.fc_rnn_hidden_embedded_obs(
            torch.cat([rnn_hidden, embedded_obs], dim=1)))
        mean = self.fc_state_mean_posterior(hidden)
        stddev = F.softplus(self.fc_state_stddev_posterior(hidden)) + self._min_stddev
        return Normal(mean, stddev)
