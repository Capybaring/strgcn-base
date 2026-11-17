"""Official variational mTAN encoder/decoder, exposing pre-output features."""

import torch
from torch import nn

from .vendor.models import dec_mtan_rnn, enc_mtan_rnn


class MTANCore(nn.Module):
    def __init__(
        self,
        num_vars,
        hid_dim=64,
        latent_dim=32,
        gen_hidden=50,
        te_dim=128,
        num_ref_points=64,
        n_heads=1,
        device="cpu",
    ):
        super().__init__()
        if (
            min(
                num_vars,
                hid_dim,
                latent_dim,
                gen_hidden,
                te_dim,
                num_ref_points,
                n_heads,
            )
            < 1
        ):
            raise ValueError("mTAN dimensions must be positive")
        if te_dim % n_heads:
            raise ValueError("mTAN time embedding must be divisible by n_heads")
        self.latent_dim = latent_dim
        self.register_buffer("reference_times", torch.linspace(0, 1, num_ref_points))
        self.encoder = enc_mtan_rnn(
            num_vars,
            self.reference_times,
            latent_dim,
            hid_dim,
            te_dim,
            n_heads,
            learn_emb=True,
            device=device,
        )
        self.decoder = dec_mtan_rnn(
            num_vars,
            self.reference_times,
            latent_dim,
            gen_hidden,
            te_dim,
            n_heads,
            learn_emb=True,
            device=device,
        )
        self.query_dim = 2 * gen_hidden
        self.kl = None

    def encode(self, values, times, mask, sample):
        # Upstream keeps a device attribute and an ordinary reference tensor.
        # Refresh these from the module so later .to(device) calls remain safe.
        for module in (self.encoder, self.decoder):
            module.device = values.device
            module.query = self.reference_times
        values = torch.where(mask.bool(), values, 0.0)
        observed = torch.cat([values, mask.to(values.dtype)], -1)
        parameters = self.encoder(observed, times)
        mu, logvar = parameters.split(self.latent_dim, dim=-1)
        self.kl = 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(-1).mean()
        return mu + torch.randn_like(mu) * (0.5 * logvar).exp() if sample else mu

    def query_features(self, z, query_times):
        # Exactly decoder.forward up to (but excluding) z0_to_obs.
        hidden, _ = self.decoder.gru_rnn(z)
        query = self.decoder.learn_time_embedding(query_times)
        key = self.decoder.learn_time_embedding(self.reference_times.unsqueeze(0))
        return self.decoder.att(query, key, hidden)

    def forward(self, values, times, mask, query_times, sample=False):
        z = self.encode(values, times, mask, sample)
        return self.decoder.z0_to_obs(self.query_features(z, query_times))
