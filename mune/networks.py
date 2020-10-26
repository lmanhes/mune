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
    def __init__(self, determ_state_dim, stoch_state_dim, out_features):
        super().__init__()
        self.fc = nn.Linear(determ_state_dim + stoch_state_dim, out_features)

    def forward(self, determ_state, stoch_state):
        return self.fc(torch.cat([determ_state, stoch_state], -1))


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
    def __init__(self, determ_state_dim, stoch_state_dim):
        super().__init__()

        self.fc_deter_stoch = nn.Linear(determ_state_dim + stoch_state_dim, 1024)

        self.convt_decoder = nn.Sequential(OrderedDict([
            ('convt1', nn.ConvTranspose2d(1024, 128, 5, stride=2)),
            ('relu1', nn.ReLU()),
            ('convt2', nn.ConvTranspose2d(128, 64, 5, stride=2)),
            ('relu2', nn.ReLU()),
            ('convt3', nn.ConvTranspose2d(64, 32, 6, stride=2)),
            ('relu3', nn.ReLU()),
            ('convt4', nn.ConvTranspose2d(32, 3, 6, stride=2))
        ]))

    def forward(self, determ_state, stoch_state):
        x = self.fc_deter_stoch(torch.cat([determ_state, stoch_state], -1))
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
    """
    def __init__(self, emb_dim, action_dim, rnn_hidden_dim=200,
                 hidden_dim=200, stoch_dim=30, min_stddev=0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.action_dim = action_dim
        self.rnn_hidden_dim = rnn_hidden_dim

        # deterministic
        self.fc_state_action = nn.Linear(stoch_dim + action_dim, rnn_hidden_dim)

        # prior stochastic
        #self.fc_prior = nn.Linear(rnn_hidden_dim, hidden_dim)
        #self.fc_prior_ms = nn.Linear(hidden_dim, 2*stoch_dim)

        # posterior stochastic
        self.fc_posterior = nn.Linear(rnn_hidden_dim+emb_dim, hidden_dim)
        self.fc_posterior_ms = nn.Linear(hidden_dim, 2*stoch_dim)

        self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)

        self.min_stddev = min_stddev

    def deterministic_state(self, state, action, rnn_hidden):
        """
        h_t+1 = f(h_t, s_t, a_t)
        """
        state_action = F.relu(self.fc_state_action(torch.cat([state, action], dim=1)))
        return self.rnn(state_action, rnn_hidden)

    def stochastic_state_prior(self, determ_state):
        """
        s_t ~ p(s_t | h_t)
        """
        hidden = F.relu(self.fc_prior(determ_state))
        mean, stddev = torch.chunk(self.fc_prior_ms(hidden), 2, dim=-1)
        stddev = F.softplus(stddev) + self.min_stddev
        return Normal(mean, stddev)

    def stochastic_state_posterior(self, rnn_hidden, embed_state):
        """
        s'_t ~ p(s'_t | (e_t, h_t))
        """
        hidden = F.relu(self.fc_posterior(
            torch.cat([rnn_hidden, embed_state], dim=1)))
        mean, stddev = torch.chunk(self.fc_posterior_ms(hidden), 2, dim=-1)
        stddev = F.softplus(stddev) + self.min_stddev
        stochastic_state = Normal(mean, stddev).rsample()
        return mean, stddev, stochastic_state

    def forward(self, embed_state, prev_action, prev_post_state, prev_hidden_state):
        determ_state = self.deterministic_state(prev_post_state, prev_action, prev_hidden_state)
        mean, stddev, stoch_state = self.stochastic_state_posterior(determ_state, embed_state)
        return {
            "determ_state": determ_state,
            "stoch_state": stoch_state,
            "mean": mean,
            "stddev": stddev
        }
