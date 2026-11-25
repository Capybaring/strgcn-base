"""Official model operators with pre-output query interfaces (no X fusion)."""

from types import SimpleNamespace

import torch
from torch import nn


class GraFITiCore(nn.Module):
    def __init__(self, num_vars, hid_dim=64, n_layer=4, n_heads=2):
        super().__init__()
        from .vendor.grafiti.grafiti_layers import grafiti_

        if hid_dim % n_heads:
            raise ValueError("GraFITi width must be divisible by heads")
        self.model = grafiti_(num_vars, hid_dim, n_layer, n_heads, device="cpu")
        self.query_dim = 3 * hid_dim

    @property
    def output_layer(self):
        return self.model.output

    def query_features(self, values, times, mask, queries, sample=False):
        q, n = len(queries), values.shape[-1]
        all_times = torch.cat([times, queries])[None]
        all_values = torch.cat([torch.where(mask, values, 0), values.new_zeros(q, n)])[
            None
        ]
        obs = torch.cat([mask, mask.new_zeros(q, n)])[None].to(values.dtype)
        target = torch.cat([mask.new_zeros(mask.shape), mask.new_ones(q, n)])[None].to(
            values.dtype
        )
        return self.model(all_times, all_values, obs, target, return_features=True)[
            0, len(times) :
        ]


class LatentODECore(nn.Module):
    def __init__(
        self,
        num_vars,
        hid_dim=40,
        latent_dim=20,
        ode_hidden=100,
        rec_layers=1,
        gen_layers=1,
        gru_units=100,
        ode_rtol=1e-3,
        ode_atol=1e-4,
    ):
        super().__init__()
        from .vendor.latent_ode import utils
        from .vendor.latent_ode.encoder_decoder import Encoder_z0_ODE_RNN, Decoder
        from .vendor.latent_ode.ode_func import ODEFunc
        from .vendor.latent_ode.diffeq_solver import DiffeqSolver

        rec_func = ODEFunc(
            2 * num_vars,
            hid_dim,
            utils.create_net(hid_dim, hid_dim, rec_layers, ode_hidden),
        )
        rec_solver = DiffeqSolver(
            2 * num_vars,
            rec_func,
            "euler",
            hid_dim,
            odeint_rtol=ode_rtol,
            odeint_atol=ode_atol,
        )
        self.encoder = Encoder_z0_ODE_RNN(
            hid_dim, 2 * num_vars, rec_solver, z0_dim=latent_dim, n_gru_units=gru_units
        )
        gen_func = ODEFunc(
            num_vars,
            latent_dim,
            utils.create_net(latent_dim, latent_dim, gen_layers, ode_hidden),
        )
        self.solver = DiffeqSolver(
            num_vars,
            gen_func,
            "dopri5",
            latent_dim,
            odeint_rtol=ode_rtol,
            odeint_atol=ode_atol,
        )
        self.decoder = Decoder(latent_dim, num_vars)
        self.query_dim, self.kl = latent_dim, None

    @property
    def output_layer(self):
        return self.decoder.decoder[0]

    def query_features(self, values, times, mask, queries, sample=False):
        self.encoder.device = values.device
        observed = torch.cat([torch.where(mask, values, 0), mask.to(values.dtype)], -1)[
            None
        ]
        mu, std = self.encoder(observed, times)
        std = std.clamp_min(1e-5)
        self.kl = 0.5 * (mu.square() + std.square() - 1 - 2 * std.log()).sum(-1).mean()
        z0 = mu + torch.randn_like(mu) * std if sample else mu
        # z0 is anchored at the first history time, not at the first query time.
        timeline, inverse = torch.cat([times[:1], queries]).unique(
            sorted=True, return_inverse=True
        )
        trajectory = self.solver(z0, timeline)[0, 0]
        return trajectory[inverse[1:]]


def _vector_hidden(input_dim, hidden, depth):
    layers = []
    for _ in range(depth):
        layers.extend([nn.Linear(input_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden)])
        input_dim = hidden
    return nn.ModuleList(layers), hidden


class CRUCore(nn.Module):
    """Full CRU/Van Loan transitions and original Physionet/USHCN decoder."""

    def __init__(self, num_vars, hid_dim=50, latent_dim=20, num_basis=15, bandwidth=3):
        super().__init__()
        from .vendor.cru.encoder import Encoder
        from .vendor.cru.decoder import SplitDiagGaussianDecoder
        from .vendor.cru.CRULayer import CRULayer
        from .vendor.cru.CRUCell import var_activation_inverse

        if latent_dim < 2 or latent_dim % 2:
            raise ValueError("CRU latent_dim must be positive and even")
        lod = latent_dim // 2

        class VectorEncoder(Encoder):
            def _build_hidden_layers(self):
                return _vector_hidden(num_vars, hid_dim, 3)

        class VectorDecoder(SplitDiagGaussianDecoder):
            def _build_hidden_layers_mean(self):
                return _vector_hidden(latent_dim, hid_dim, 3)

            def _build_hidden_layers_var(self):
                return _vector_hidden(3 * lod, hid_dim, 1)

        args = SimpleNamespace(
            rkn=False,
            f_cru=False,
            orthogonal=False,
            num_basis=num_basis,
            bandwidth=min(bandwidth, lod - 1),
            trans_var_activation="elup1",
            trans_covar=0.1,
            trans_net_hidden_units=[],
            trans_net_hidden_activation="tanh",
            t_sensitive_trans_net=False,
        )
        self.encoder = VectorEncoder(lod, "elup1", "pre").double()
        self.layer = CRULayer(lod, args).double()
        self.decoder = VectorDecoder(lod, num_vars, "elup1").double()
        self.register_buffer(
            "initial_mean", torch.zeros(1, latent_dim, dtype=torch.double)
        )
        self.register_buffer("initial_side", torch.zeros(1, lod, dtype=torch.double))
        initial = float(var_activation_inverse(10.0))
        self.initial_log_upper = nn.Parameter(
            torch.full((1, lod), initial, dtype=torch.double)
        )
        self.initial_log_lower = nn.Parameter(
            torch.full((1, lod), initial, dtype=torch.double)
        )
        self.query_dim, self.forecast_variance = latent_dim, None

    @property
    def output_layer(self):
        return self.decoder._out_layer_mean

    def query_features(self, values, times, mask, queries, sample=False):
        from .vendor.cru.CRUCell import var_activation

        dtype = self.initial_mean.dtype
        values, times, queries = values.to(dtype), times.to(dtype), queries.to(dtype)
        q, n = len(queries), values.shape[-1]
        timeline = torch.cat([times, queries])[None]
        if (timeline.diff(dim=-1) < 0).any():
            raise ValueError("CRU forecast queries must follow sorted history")
        observed = torch.cat([torch.where(mask, values, 0), values.new_zeros(q, n)])
        y, variance = self.encoder(observed)
        valid = torch.cat([mask.any(-1), mask.new_zeros(q)])[None]
        post, cov, _, _, _ = self.layer(
            y[None],
            variance[None],
            self.initial_mean,
            [
                var_activation(self.initial_log_upper),
                var_activation(self.initial_log_lower),
                self.initial_side,
            ],
            obs_valid=valid,
            time_points=timeline,
        )
        mean, covariance = post[0, len(times) :], torch.cat(cov, -1)[0, len(times) :]
        # Preserve the complete upstream variance decoder and its positive output.
        _, self.forecast_variance = self.decoder(mean, covariance)
        return mean
