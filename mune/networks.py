import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Normal, OneHotCategorical
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
    def __init__(self, hybrid_state_dim, out_features):
        super().__init__()
        self.fc = nn.Linear(hybrid_state_dim, out_features)

    def forward(self, hybrid_state):
        return self.fc(hybrid_state)


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
    def __init__(self, hybrid_state_dim):
        super().__init__()

        self.fc_deter_stoch = nn.Linear(hybrid_state_dim, 1024)

        self.convt_decoder = nn.Sequential(OrderedDict([
            ('convt1', nn.ConvTranspose2d(1024, 128, 5, stride=2)),
            ('relu1', nn.ReLU()),
            ('convt2', nn.ConvTranspose2d(128, 64, 5, stride=2)),
            ('relu2', nn.ReLU()),
            ('convt3', nn.ConvTranspose2d(64, 32, 6, stride=2)),
            ('relu3', nn.ReLU()),
            ('convt4', nn.ConvTranspose2d(32, 3, 6, stride=2))
        ]))

    def forward(self, hybrid_state):
        x = self.fc_deter_stoch(hybrid_state)
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
        self.fc_prior = nn.Linear(rnn_hidden_dim, hidden_dim)
        self.fc_prior_ms = nn.Linear(hidden_dim, 2*stoch_dim)

        # posterior stochastic
        self.fc_posterior = nn.Linear(rnn_hidden_dim+emb_dim, hidden_dim)
        self.fc_posterior_ms = nn.Linear(hidden_dim, 2*stoch_dim)

        self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)

        self.min_stddev = min_stddev

    def deterministic_state(self, prev_stoch_state, prev_action, prev_determ_state):
        """
        h_t+1 = f(h_t, s_t, a_t)
        """
        state_action = F.relu(self.fc_state_action(torch.cat([prev_stoch_state, prev_action], dim=1)))
        return self.rnn(state_action, prev_determ_state)

    def stochastic_state_prior(self, determ_state):
        """
        s_t ~ p(s_t | h_t)
        """
        hidden = F.relu(self.fc_prior(determ_state))
        mean, stddev = torch.chunk(self.fc_prior_ms(hidden), 2, dim=-1)
        stddev = F.softplus(stddev) + self.min_stddev
        stoch_sate =  Normal(mean, stddev)
        return {
            "stoch_state": stoch_sate,
            "mean": mean,
            "stddev": stddev
        }

    def stochastic_state_posterior(self, determ_state, embed_state):
        """
        s'_t ~ p(s'_t | (e_t, h_t))
        """
        hidden = F.relu(self.fc_posterior(
            torch.cat([determ_state, embed_state], dim=1)))
        mean, stddev = torch.chunk(self.fc_posterior_ms(hidden), 2, dim=-1)
        stddev = F.softplus(stddev) + self.min_stddev
        stoch_state = Normal(mean, stddev).rsample()
        return {
            "stoch_state": stoch_state,
            "mean": mean,
            "stddev": stddev
        }

    def forward(self, embed_state, prev_action, prev_stoch_state, prev_determ_state):
        determ_state = self.deterministic_state(prev_stoch_state, prev_action, prev_determ_state)
        stoch_state_prior = self.stochastic_state_prior(determ_state)
        stoch_state_posterior = self.stochastic_state_posterior(determ_state, embed_state)
        hybrid_state = torch.cat([determ_state, stoch_state_posterior['stoch_state']], dim=-1)
        return {
            "determ_state": determ_state,
            "stoch_state_prior": stoch_state_prior,
            "stoch_state_posterior": stoch_state_posterior,
            "hybrid_state": hybrid_state
        }


class DenseModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1, act=nn.ReLU):
        super().__init__()
        self.act = act
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        self.model = self._build()

    def _build(self):
        layers = [nn.Linear(self.input_dim, self.hidden_dim)]
        for l in range(self.n_layers-1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(self.act())
        layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ActionDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, act=nn.ReLU):
        super().__init__()

        self.model = DenseModel(input_dim=input_dim,
                                hidden_dim=hidden_dim,
                                output_dim=output_dim,
                                n_layers=n_layers,
                                act=act)

    def forward(self, x):
        x = self.model(x)
        return OneHotCategorical(logits=x)


class AgentModel(nn.Module):
    def __init__(self,
                 action_output_dim,
                 modalities_config: dict,
                 determ_state_dim=200,
                 stoch_state_dim=30,
                 min_stddev=0.1,
                 reward_hidden_dim=100,
                 reward_n_layers=2,
                 value_hidden_dim=100,
                 value_n_layers=2,
                 action_hidden_dim=100,
                 action_n_layers=2):
        super().__init__()
        self.modalities_config = modalities_config
        hybrid_state_dim = determ_state_dim + stoch_state_dim

        self.encoder_modalities = nn.ModuleDict()
        self.decoder_modalities = nn.ModuleDict()
        for m_name, m in modalities_config.items():
            if m['type'] == "proprio":
                encoder = ProprioEncoder(in_features=m['in_features'])
                self.encoder_modalities.add_module(m_name, encoder)
                decoder = ProprioDecoder(hybrid_state_dim=hybrid_state_dim,
                                         out_features=m['in_features'])
                self.decoder_modalities.add_module(m_name, decoder)
            elif m['type'] == "vision":
                encoder = VisionEncoder()
                self.encoder_modalities.add_module(m_name, encoder)
                decoder = VisionDecoder(hybrid_state_dim=hybrid_state_dim)
                self.decoder_modalities.add_module(m_name, decoder)

        embed_dim = sum([m.output_size for name, m in self.encoder_modalities.items()])
        self.fusion_layer = FusionLayer(in_features=embed_dim)

        self.rssm = Rssm(emb_dim=self.fusion_layer.output_size,
                         action_dim=action_output_dim,
                         rnn_hidden_dim=determ_state_dim,
                         hidden_dim=determ_state_dim,
                         stoch_dim=stoch_state_dim,
                         min_stddev=min_stddev)

        self.reward_model = DenseModel(input_dim=hybrid_state_dim,
                                       hidden_dim=reward_hidden_dim,
                                       n_layers=reward_n_layers,
                                       output_dim=1)

        self.value_model = DenseModel(input_dim=hybrid_state_dim,
                                      hidden_dim=value_hidden_dim,
                                      n_layers=value_n_layers,
                                      output_dim=1)

        self.action_decoder = ActionDecoder(input_dim=hybrid_state_dim,
                                            hidden_dim=action_hidden_dim,
                                            n_layers=action_n_layers,
                                            output_dim=action_output_dim)

    def encode_observations(self, observations: dict):
        embeddings = []
        for name, value in observations.items():
            encoder = self.encoder_modalities[name]
            x = encoder(value)
            embeddings.append(x)
        return self.fusion_layer(torch.cat(embeddings, dim=-1))

    def decode_observations(self, hybrid_state):
        observations = {}
        for m_name in self.modalities_config:
            decoder = self.decoder_modalities[m_name]
            x = decoder(hybrid_state)
            observations[m_name] = x
        return observations

    def forward(self, observations: dict, prev_action, prev_determ_state, prev_stoch_state):
        embed_state = self.encode_observations(observations)

        transitions = self.rssm(embed_state=embed_state,
                                prev_action=prev_action,
                                prev_stoch_state=prev_stoch_state,
                                prev_determ_state=prev_determ_state)
        hybrid_state = transitions["hybrid_state"]

        return {
            "pred_reward": self.reward_model(hybrid_state),
            "pred_action": self.action_decoder(hybrid_state),
            "pred_value": self.value_model(hybrid_state),
            "pred_observations": self.decode_observations(hybrid_state),
            "transitions": transitions
        }
