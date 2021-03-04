import torch
import torch.nn as nn
import torch.nn.functional as F

from mune.networks import (
    ProprioEncoder, ProprioDecoder,
    VisionEncoder, VisionDecoder,
    FusionLayer
)


class MultimodalBetaVAEConfig:

    modalities = {
        "vision": {
            "type": "vision"
        },
        "proprio": {
            "type": "proprio",
            "in_features": 4
        }
    }

    fusion_state_dim = 256
    latent_dim = 128
    beta = 5

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class MultimodalBetaVAE(nn.Module):

    def __init__(self, config: MultimodalBetaVAEConfig):
        super().__init__()
        self.config = config
        self.encoder_modalities = nn.ModuleDict()
        self.decoder_modalities = nn.ModuleDict()
        for m_name, m in config.modalities.items():
            if m['type'] == "proprio":
                encoder = ProprioEncoder(in_features=m['in_features'])
                self.encoder_modalities.add_module(m_name, encoder)
                decoder = ProprioDecoder(in_features=config.fusion_state_dim,
                                         out_features=m['in_features'])
                self.decoder_modalities.add_module(m_name, decoder)
            elif m['type'] == "vision":
                encoder = VisionEncoder()
                self.encoder_modalities.add_module(m_name, encoder)
                decoder = VisionDecoder(in_features=config.fusion_state_dim)
                self.decoder_modalities.add_module(m_name, decoder)

        merge_dim = sum([m.output_size for name, m in self.encoder_modalities.items()])
        self.fusion_layer = FusionLayer(in_features=merge_dim,
                                        out_features=config.fusion_state_dim)

        self.fc_mu = nn.Linear(config.fusion_state_dim, config.latent_dim)
        self.fc_var = nn.Linear(config.fusion_state_dim, config.latent_dim)

        self.z_to_fusion_state = nn.Linear(config.latent_dim, config.fusion_state_dim)

    def encode(self, observations: dict):
        embeddings = []
        for name, value in observations.items():
            encoder = self.encoder_modalities[name]
            x = encoder(value)
            embeddings.append(x)
        fusion_state = self.fusion_layer(torch.cat(embeddings, dim=-1))

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(fusion_state)
        log_var = self.fc_var(fusion_state)

        return fusion_state, mu, log_var

    def decode(self, z):
        fusion_state = self.z_to_fusion_state(z)

        observations = {}
        for m_name in self.config.modalities:
            decoder = self.decoder_modalities[m_name]
            x = decoder(fusion_state)
            observations[m_name] = x
        return observations

    def reparameterize(self, mu, logvar):
        """
        Will a single z be enough to compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def loss(self,
             observations: dict,
             mu,
             log_var,
             reco_observations: dict,
             ):
        #self.num_iter += 1
        #kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        reco_loss = []
        for modality in observations:
            weight = 0.9 if modality == "vision"  else 0.1
            reco_loss.append(weight * F.mse_loss(reco_observations[modality], observations[modality]))
        reco_loss = torch.mean(torch.stack(reco_loss), dim=0)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = reco_loss + self.config.beta * kld_loss

        return {
            'loss': loss,
            'reco_loss': reco_loss,
            'kld_loss': kld_loss
        }

    def forward(self, observations: dict):
        fusion_state, mu, log_var = self.encode(observations)
        z = self.reparameterize(mu, log_var)
        reco_observations = self.decode(z)
        return mu, log_var, z, reco_observations