"""
================================================================================
STVAE: Multi-resolution Deconvolution for Spatial Transcriptomics
================================================================================

COMPLETE STANDALONE IMPLEMENTATION


Key Features Implemented:
1. scVAE: Conditional VAE for scRNA-seq reference
2. STVAE (stVAE): Spatial deconvolution with continuous cell states
3. Empirical prior learning from scRNA-seq latents
4. Amortized and non-amortized inference for gamma
5. Cell-type proportion estimation with unknown cell type
6. Gene-specific correction factors (alpha)
7. Cell-type-specific expression imputation
8. Differential expression analysis utilities
9. Spatial smoothing and visualization utilities

Mathematical Framework:
- scRNA-seq: x_n ~ NB(l_n * ρ(γ_n, c_n), θ), γ_n ~ N(0, I)
- Spatial: x_s ~ NB(l_s * α ⊙ Σ_c π_sc * ρ(γ_sc, c), θ)
- Prior: γ_sc ~ N(μ_c, Σ_c) learned from scRNA-seq

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.distributions import Normal, NegativeBinomial, kl_divergence
import numpy as np
from typing import Tuple, Optional, Dict, List, Union, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from abc import ABC, abstractmethod
import warnings
import math
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cdist
import logging
import argparse
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Must be BEFORE importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# Try importing UMAP, fall back to TSNE if not available
try:
    import umap.umap_ as umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# ==============================================================================
# GLOBAL MATPLOTLIB FONT SETTINGS
# ==============================================================================
plt.rcParams.update({
    # Font family
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    
    # Global defaults
    'font.size': 16,
    'font.weight': 'bold',
    
    # Axes settings
    'axes.titlesize': 18,
    'axes.titleweight': 'bold',
    'axes.labelsize': 16,
    'axes.labelweight': 'bold',
    'axes.linewidth': 1.5,
    
    # Tick settings
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    
    # Legend settings
    'legend.fontsize': 14,
    'legend.title_fontsize': 15,
    
    # Figure settings
    'figure.titlesize': 20,
    'figure.titleweight': 'bold',
    
    # Savefig settings
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})
# Global variable for hexagon orientation (set via command line)
HEXAGON_ORIENTATION = np.radians(0)  # Default: pointy-top
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==============================================================================
# SECTION 1: CONFIGURATION AND DATA STRUCTURES
# ==============================================================================

@dataclass
class STVAEConfig:
    """
    Comprehensive configuration for STVAE model.
    
    Contains all hyperparameters for both scVAE and stVAE stages.
    
    Two-Stage Training Pipeline:
    ---------------------------
    Stage 1: scVAE training on single-cell reference
        - Learns decoder p(x|γ,c) and dispersion θ
        - Fits empirical prior p(γ|c) from latent representations
    
    Stage 2a: stAE (proportion/intensity estimation)
        - Uses FIXED reference profiles (mean expression per cell type)
        - Learns: π (proportions), α (gene correction factor)
        - Outputs: proportions.csv, intensity_maps.csv, alpha.csv
        - These are the FINAL deconvolution outputs
    
    Stage 2b: stVAE (cell state inference)
        - Uses FROZEN scVAE decoder
        - Optionally freezes π from Stage 2a (--freeze_proportions)
        - Optionally freezes α from Stage 2a (--freeze_intensity)
        - Learns: γ (cell-type-specific states per spot)
        - γ captures continuous phenotypic variation within cell types
    
    Key Parameters:
    --------------
    - α (alpha): Gene-specific correction factor that accounts for:
        * Platform differences between scRNA-seq and spatial
        * Gene capture efficiency variations
        * Technical sensitivity differences
        * Constrained to [alpha_min, alpha_max]
    
    - Intensity: DERIVED value, not directly learned:
        intensity_c = library_size × π_c × (α · profile_c).sum()
        Represents the expected total counts from cell type c at each spot
    """
    # Data dimensions (will be updated during fitting)
    n_genes: int = 2000
    n_cell_types: int = 10
    n_cells: int = 5000
    n_spots: int = 1000
    
    # Latent space dimensions
    n_latent: int = 10                    # γ dimension
    n_layers_encoder: int = 2             # Encoder depth
    n_layers_decoder: int = 2             # Decoder depth  
    n_hidden: int = 256                   # Hidden layer size
    
    # Cell type embedding
    n_cell_type_embedding: int = 16       # Cell type embedding dimension
    
    # Dispersion modeling
    dispersion: str = 'gene'              # 'gene', 'gene-cell', 'gene-batch'
    gene_likelihood: str = 'nb'           # 'nb' or 'zinb'
    
    # scVAE training
    sc_learning_rate: float = 1e-3
    sc_weight_decay: float = 1e-6
    sc_max_epochs: int = 400
    sc_batch_size: int = 128
    sc_early_stopping: bool = True
    sc_early_stopping_patience: int = 30
    sc_kl_weight: float = 1.0
    
    # stVAE training  
    st_learning_rate: float = 1e-3
    st_weight_decay: float = 1e-6
    st_max_epochs: int = 2500
    st_batch_size: int = 128
    st_early_stopping: bool = True
    st_early_stopping_patience: int = 50
    
    # Regularization
    lambda_reg: float = 1e-4              # L2 regularization
    dropout_rate: float = 0.1             # Dropout rate
    
    # Gamma inference
    amortized_gamma: bool = True          # Use amortized inference for γ
    inference_mode: str = 'amortized'     # ADD THIS LINE: 'amortized' or 'non_amortized'

    # Prior settings
    prior_type: str = 'gaussian'          # 'gaussian' or 'gmm'
    n_prior_components: int = 1           # GMM components per cell type
    
    # Proportion estimation
    add_unknown_cell_type: bool = False   # Keep only ONE definition
    proportion_temperature: float = 0.5
    lambda_pi_entropy: float = 10.0        # Entropy regularization on π
    lambda_pi_sparsity: float = 1.0       # Sparsity regularization on π

    # Gene correction
    lambda_alpha: float = 1e-2            # α regularization weight
    alpha_min: float = 0.1                # Minimum α value
    alpha_max: float = 10.0               # Maximum α value
    
    # Gamma regularization
    lambda_gamma: float = 1.0             # γ prior penalty weight
    
    # Cell type balancing
    use_cell_type_weights: bool = True    # Balance cell types in training
    cell_type_weight_cap: float = 0.05    # Minimum proportion for weighting
    
    # Advanced options
    use_batch_norm: bool = False          # Use batch normalization
    use_layer_norm: bool = True           # Use layer normalization
    activation: str = 'relu'              # Activation function
    
    # Imputation settings
    n_samples_imputation: int = 25        # Samples for imputation uncertainty

    # --- NEW: Pseudo-spot Consistency Regularization ---
    use_pseudo_spots: bool = True     # Enable this feature
    n_pseudo_spots: int = 1000        # Number of synthetic spots to maintain
    cells_per_spot_range: List[int] = field(default_factory=lambda: [5, 15]) 
    pseudo_weight: float = 1.0        # Weight of the pseudo-spot loss
    pseudo_warmup_epochs: int = 50    # Slowly increase weight
    pseudo_training_ratio: int = 1    # ADD THIS LINE - Ratio of pseudo batch size to real batch size
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.n_latent > 0, "n_latent must be positive"
        assert self.dispersion in ['gene', 'gene-cell', 'gene-batch']
        assert self.gene_likelihood in ['nb', 'zinb']
        assert self.prior_type in ['gaussian', 'gmm']


@dataclass
class TrainingState:
    """Tracks training progress and metrics."""
    epoch: int = 0
    best_loss: float = float('inf')
    patience_counter: int = 0
    history: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    converged: bool = False


# ==============================================================================
# SECTION 2: DISTRIBUTION UTILITIES
# ==============================================================================

class DistributionUtils:
    """Utility functions for probability distributions."""
    
    @staticmethod
    def log_nb_positive(
        x: torch.Tensor,
        mu: torch.Tensor,
        theta: torch.Tensor,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Log probability of Negative Binomial distribution (NB2 parameterization).
        
        NB(x; μ, θ) where:
        - μ is the mean
        - θ is the inverse dispersion (larger θ = less dispersion)
        
        Args:
            x: Observed counts [batch, genes]
            mu: Mean parameter [batch, genes]
            theta: Dispersion parameter [genes] or [batch, genes]
            eps: Numerical stability constant
            
        Returns:
            Log probability [batch, genes]
        """
        x = x.float()
        mu = mu.float().clamp(min=eps)
        theta = theta.float().clamp(min=eps)
        
        if theta.dim() == 1:
            theta = theta.unsqueeze(0).expand_as(mu)
        
        # Log probability computation
        log_theta_mu = torch.log(theta + mu + eps)
        
        log_prob = (
            torch.lgamma(x + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x + 1.0)
            + theta * (torch.log(theta + eps) - log_theta_mu)
            + x * (torch.log(mu + eps) - log_theta_mu)
        )
        
        return log_prob
    
    @staticmethod
    def log_zinb_positive(
        x: torch.Tensor,
        mu: torch.Tensor,
        theta: torch.Tensor,
        pi: torch.Tensor,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Log probability of Zero-Inflated Negative Binomial distribution.
        
        Args:
            x: Observed counts
            mu: Mean of NB component
            theta: Dispersion of NB component
            pi: Dropout probability (probability of extra zeros)
            
        Returns:
            Log probability
        """
        # Softplus to ensure pi is in valid range
        pi = pi.clamp(min=eps, max=1 - eps)
        
        # NB log probability
        log_nb = DistributionUtils.log_nb_positive(x, mu, theta, eps)
        
        # Case: x = 0
        theta_safe = theta.clamp(min=eps)
        log_nb_zero = theta * (torch.log(theta_safe) - torch.log(theta + mu + eps))
        
        # Zero case: log(π + (1-π) * NB(0))
        log_zero = torch.log(pi + (1 - pi) * torch.exp(log_nb_zero) + eps)
        
        # Non-zero case: log((1-π) * NB(x))
        log_nonzero = torch.log(1 - pi + eps) + log_nb
        
        # Combine
        is_zero = (x < eps).float()
        log_prob = is_zero * log_zero + (1 - is_zero) * log_nonzero
        
        return log_prob
    
    @staticmethod
    def kl_divergence_gaussian(
        q_mu: torch.Tensor,
        q_logvar: torch.Tensor,
        p_mu: Optional[torch.Tensor] = None,
        p_logvar: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        KL divergence between two diagonal Gaussians: KL(q || p).
        
        If p_mu and p_logvar are None, uses standard normal prior.
        """
        if p_mu is None and p_logvar is None:
            # KL(q || N(0, I))
            kl = 0.5 * torch.sum(
                torch.exp(q_logvar) + q_mu.pow(2) - 1.0 - q_logvar,
                dim=-1
            )
        else:
            if p_mu is None:
                p_mu = torch.zeros_like(q_mu)
            if p_logvar is None:
                p_logvar = torch.zeros_like(q_logvar)
            
            var_ratio = torch.exp(q_logvar - p_logvar)
            diff_sq = (q_mu - p_mu).pow(2) / torch.exp(p_logvar)
            
            kl = 0.5 * torch.sum(
                var_ratio + diff_sq - 1.0 - q_logvar + p_logvar,
                dim=-1
            )
        
        return kl
    
    @staticmethod
    def reparameterize(
        mu: torch.Tensor,
        logvar: torch.Tensor,
        n_samples: int = 1
    ) -> torch.Tensor:
        """
        Reparameterization trick for Gaussian.
        
        Args:
            mu: Mean [batch, dim]
            logvar: Log variance [batch, dim]
            n_samples: Number of samples
            
        Returns:
            Samples [n_samples, batch, dim] if n_samples > 1, else [batch, dim]
        """
        std = torch.exp(0.5 * logvar)
        
        if n_samples == 1:
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # [n_samples, batch, dim]
            eps = torch.randn(n_samples, *std.shape, device=std.device)
            return mu.unsqueeze(0) + eps * std.unsqueeze(0)


# ==============================================================================
# SECTION 3: NEURAL NETWORK BUILDING BLOCKS
# ==============================================================================

def get_activation(name: str) -> nn.Module:
    """Get activation function by name."""
    activations = {
        'relu': nn.ReLU(),
        'leaky_relu': nn.LeakyReLU(0.2),
        'elu': nn.ELU(),
        'gelu': nn.GELU(),
        'selu': nn.SELU(),
        'softplus': nn.Softplus(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid()
    }
    return activations.get(name.lower(), nn.ReLU())


class FCLayer(nn.Module):
    """
    Fully connected layer with optional normalization, activation, and dropout.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        activation: str = 'relu',
        dropout_rate: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        
        layers = [nn.Linear(in_features, out_features, bias=bias)]
        
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_features))
        elif use_layer_norm:
            layers.append(nn.LayerNorm(out_features))
        
        if activation != 'none':
            layers.append(get_activation(activation))
        
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MLP(nn.Module):
    """
    Multi-layer perceptron with flexible architecture.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int = 256,
        n_layers: int = 2,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        activation: str = 'relu',
        dropout_rate: float = 0.1,
        output_activation: str = 'none'
    ):
        super().__init__()
        
        layers = []
        
        # Input layer
        layers.append(FCLayer(
            in_features, hidden_features,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            activation=activation,
            dropout_rate=dropout_rate
        ))
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(FCLayer(
                hidden_features, hidden_features,
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
                activation=activation,
                dropout_rate=dropout_rate
            ))
        
        # Output layer
        layers.append(FCLayer(
            hidden_features, out_features,
            use_batch_norm=False,
            use_layer_norm=False,
            activation=output_activation,
            dropout_rate=0.0
        ))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class GaussianEncoder(nn.Module):
    """
    Encoder that outputs parameters of a Gaussian distribution.
    """
    def __init__(
        self,
        in_features: int,
        latent_dim: int,
        hidden_features: int = 256,
        n_layers: int = 2,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        activation: str = 'relu',
        dropout_rate: float = 0.1,
        var_eps: float = 1e-4,
        var_activation: str = 'softplus'
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.var_eps = var_eps
        self.var_activation = var_activation
        
        # Shared encoder
        self.encoder = MLP(
            in_features=in_features,
            out_features=hidden_features,
            hidden_features=hidden_features,
            n_layers=n_layers - 1,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            activation=activation,
            dropout_rate=dropout_rate,
            output_activation=activation
        )
        
        # Output heads
        self.mu_head = nn.Linear(hidden_features, latent_dim)
        self.var_head = nn.Linear(hidden_features, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mu: Mean of q(z|x)
            logvar: Log variance of q(z|x)
        """
        h = self.encoder(x)
        mu = self.mu_head(h)
        
        # Variance parameterization
        var_raw = self.var_head(h)
        if self.var_activation == 'softplus':
            var = F.softplus(var_raw) + self.var_eps
            logvar = torch.log(var)
        else:
            logvar = var_raw.clamp(min=-10, max=10)
        
        return mu, logvar


# ==============================================================================
# SECTION 4: scVAE - SINGLE-CELL REFERENCE MODEL
# ==============================================================================

class scVAEEncoder(nn.Module):
    """
    Encoder for scVAE: q(γ | x, c).
    
    Maps single-cell expression and cell type to latent distribution.
    """
    def __init__(self, config: STVAEConfig):
        super().__init__()
        
        self.config = config
        
        # Cell type embedding
        self.cell_type_embedding = nn.Embedding(
            config.n_cell_types, 
            config.n_cell_type_embedding
        )
        
        # Input dimension: log-normalized expression + cell type embedding
        input_dim = config.n_genes + config.n_cell_type_embedding
        
        # Gaussian encoder
        self.encoder = GaussianEncoder(
            in_features=input_dim,
            latent_dim=config.n_latent,
            hidden_features=config.n_hidden,
            n_layers=config.n_layers_encoder,
            use_batch_norm=config.use_batch_norm,
            use_layer_norm=config.use_layer_norm,
            activation=config.activation,
            dropout_rate=config.dropout_rate
        )
    
    def forward(
        self,
        x: torch.Tensor,
        cell_type_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Gene expression [batch, n_genes]
            cell_type_idx: Cell type indices [batch]
            
        Returns:
            mu: Latent mean [batch, n_latent]
            logvar: Latent log-variance [batch, n_latent]
        """
        # Log-normalize expression
        log_x = torch.log1p(x)
        
        # Get cell type embedding
        ct_emb = self.cell_type_embedding(cell_type_idx)
        
        # Concatenate inputs
        encoder_input = torch.cat([log_x, ct_emb], dim=-1)
        
        # Encode
        mu, logvar = self.encoder(encoder_input)
        
        return mu, logvar


class scVAEDecoder(nn.Module):
    """
    Decoder for scVAE: p(x | γ, c) = NB(l * ρ(γ, c), θ).
    
    Maps latent state and cell type to gene expression distribution.
    """
    def __init__(self, config: STVAEConfig):
        super().__init__()
        
        self.config = config
        
        # Cell type embedding (can be shared or separate from encoder)
        self.cell_type_embedding = nn.Embedding(
            config.n_cell_types,
            config.n_cell_type_embedding
        )
        
        # Input dimension: latent + cell type embedding
        input_dim = config.n_latent + config.n_cell_type_embedding
        
        # Decoder network outputs unnormalized log-rates
        self.decoder = MLP(
            in_features=input_dim,
            out_features=config.n_genes,
            hidden_features=config.n_hidden,
            n_layers=config.n_layers_decoder,
            use_batch_norm=config.use_batch_norm,
            use_layer_norm=config.use_layer_norm,
            activation=config.activation,
            dropout_rate=config.dropout_rate,
            output_activation='none'
        )
    
    def forward(
        self,
        z: torch.Tensor,
        cell_type_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z: Latent state [batch, n_latent]
            cell_type_idx: Cell type indices [batch]
            
        Returns:
            rho: Normalized expression profile [batch, n_genes]
        """
        # Get cell type embedding
        ct_emb = self.cell_type_embedding(cell_type_idx)
        
        # Concatenate inputs
        decoder_input = torch.cat([z, ct_emb], dim=-1)
        
        # Decode to log-rates
        log_rates = self.decoder(decoder_input)
        
        # Softmax normalization to get ρ
        rho = F.softmax(log_rates, dim=-1)
        
        return rho


class scVAE(nn.Module):
    """
    Single-cell Latent Variable Model.
    
    Conditional VAE for single-cell RNA-seq that learns:
    1. Cell-type-specific decoder ρ(γ, c)
    2. Gene-specific dispersion θ
    3. Latent representation γ for each cell
    
    Generative model:
        γ_n ~ N(0, I)
        x_ng | γ_n, c_n ~ NB(l_n * ρ_g(γ_n, c_n), θ_g)
    """
    
    def __init__(self, config: STVAEConfig):
        super().__init__()
        
        self.config = config
        
        # Encoder and decoder
        self.encoder = scVAEEncoder(config)
        self.decoder = scVAEDecoder(config)
        
        # Gene-specific dispersion (log-scale for numerical stability)
        if config.dispersion == 'gene':
            self.log_theta = nn.Parameter(torch.zeros(config.n_genes))
        elif config.dispersion == 'gene-cell':
            # Different dispersion per gene and cell type
            self.log_theta = nn.Parameter(
                torch.zeros(config.n_cell_types, config.n_genes)
            )
        else:
            self.log_theta = nn.Parameter(torch.zeros(config.n_genes))
        
        # Optional: zero-inflation for ZINB
        if config.gene_likelihood == 'zinb':
            self.dropout_decoder = MLP(
                in_features=config.n_latent + config.n_cell_type_embedding,
                out_features=config.n_genes,
                hidden_features=config.n_hidden,
                n_layers=1,
                output_activation='sigmoid'
            )
    
    @property
    def theta(self) -> torch.Tensor:
        """Gene-specific dispersion parameter."""
        return torch.exp(self.log_theta).clamp(min=1e-4, max=1e4)
    
    def encode(
        self,
        x: torch.Tensor,
        cell_type_idx: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Encode to latent distribution."""
        mu, logvar = self.encoder(x, cell_type_idx)
        return {'mu': mu, 'logvar': logvar}
    
    def decode(
        self,
        z: torch.Tensor,
        cell_type_idx: torch.Tensor,
        library_size: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Decode from latent space."""
        # Get normalized expression profile
        rho = self.decoder(z, cell_type_idx)
        
        # Scale by library size
        mu = library_size * rho
        
        # Get dispersion
        if self.config.dispersion == 'gene-cell':
            theta = self.theta[cell_type_idx]
        else:
            theta = self.theta
        
        outputs = {'mu': mu, 'rho': rho, 'theta': theta}
        
        # Optional dropout for ZINB
        if self.config.gene_likelihood == 'zinb':
            ct_emb = self.decoder.cell_type_embedding(cell_type_idx)
            dropout_input = torch.cat([z, ct_emb], dim=-1)
            outputs['dropout'] = self.dropout_decoder(dropout_input)
        
        return outputs
    
    def forward(
        self,
        x: torch.Tensor,
        cell_type_idx: torch.Tensor,
        library_size: Optional[torch.Tensor] = None,
        n_samples: int = 1
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Gene expression [batch, n_genes]
            cell_type_idx: Cell type indices [batch]
            library_size: Library sizes [batch, 1], computed if None
            n_samples: Number of latent samples
            
        Returns:
            Dictionary with all model outputs
        """
        if library_size is None:
            library_size = x.sum(dim=-1, keepdim=True)
        
        # Encode
        enc_outputs = self.encode(x, cell_type_idx)
        mu, logvar = enc_outputs['mu'], enc_outputs['logvar']
        
        # Sample latent
        z = DistributionUtils.reparameterize(mu, logvar, n_samples)
        if n_samples > 1:
            # Average over samples for output
            z_decode = z.mean(dim=0)
        else:
            z_decode = z
        
        # Decode
        dec_outputs = self.decode(z_decode, cell_type_idx, library_size)
        
        return {
            'z_mu': mu,
            'z_logvar': logvar,
            'z': z,
            **dec_outputs
        }
    
    def loss(
        self,
        x: torch.Tensor,
        cell_type_idx: torch.Tensor,
        library_size: Optional[torch.Tensor] = None,
        kl_weight: float = 1.0,
        sample_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute ELBO loss.
        
        Loss = -E_q[log p(x|z)] + kl_weight * KL(q(z|x) || p(z))
        """
        outputs = self.forward(x, cell_type_idx, library_size)
        
        # Reconstruction loss (negative log likelihood)
        if self.config.gene_likelihood == 'nb':
            log_prob = DistributionUtils.log_nb_positive(
                x, outputs['mu'], outputs['theta']
            )
        else:  # zinb
            log_prob = DistributionUtils.log_zinb_positive(
                x, outputs['mu'], outputs['theta'], outputs['dropout']
            )
        
        recon_loss = -log_prob.sum(dim=-1)  # [batch]
        
        # KL divergence (standard normal prior)
        kl_loss = DistributionUtils.kl_divergence_gaussian(
            outputs['z_mu'], outputs['z_logvar']
        )  # [batch]
        
        # Per-sample loss
        per_sample_loss = recon_loss + kl_weight * kl_loss
        
        # Apply sample weights if provided
        if sample_weights is not None:
            per_sample_loss = per_sample_loss * sample_weights
        
        # Total loss
        total_loss = per_sample_loss.mean()
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss.mean(),
            'kl_loss': kl_loss.mean(),
            'per_sample_loss': per_sample_loss
        }
    
    @torch.no_grad()
    def get_latent_representation(
        self,
        x: torch.Tensor,
        cell_type_idx: torch.Tensor,
        return_variance: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Get posterior mean of latent representation."""
        self.eval()
        enc_outputs = self.encode(x, cell_type_idx)
        
        if return_variance:
            return enc_outputs['mu'], torch.exp(enc_outputs['logvar'])
        return enc_outputs['mu']


# ==============================================================================
# SECTION 5: EMPIRICAL PRIOR FOR GAMMA
# ==============================================================================

class EmpiricalPrior(nn.Module):
    """
    Empirical prior over γ learned from scRNA-seq data.
    
    For each cell type c, learns:
    - Gaussian: p(γ|c) = N(μ_c, diag(σ²_c))
    - GMM: p(γ|c) = Σ_k π_ck * N(μ_ck, diag(σ²_ck))
    """
    
    def __init__(
        self,
        n_cell_types: int,
        n_latent: int,
        prior_type: str = 'gaussian',
        n_components: int = 1,
        eps: float = 1e-4
    ):
        super().__init__()
        
        self.n_cell_types = n_cell_types
        self.n_latent = n_latent
        self.prior_type = prior_type
        self.n_components = n_components
        self.eps = eps
        
        # Register as buffers (not parameters)
        if prior_type == 'gaussian':
            self.register_buffer('mu', torch.zeros(n_cell_types, n_latent))
            self.register_buffer('var', torch.ones(n_cell_types, n_latent))
        else:  # gmm
            self.register_buffer(
                'mu', 
                torch.zeros(n_cell_types, n_components, n_latent)
            )
            self.register_buffer(
                'var', 
                torch.ones(n_cell_types, n_components, n_latent)
            )
            self.register_buffer(
                'weights',
                torch.ones(n_cell_types, n_components) / n_components
            )
        
        self._fitted = False
    
    @torch.no_grad()
    def fit(
        self,
        z_means: torch.Tensor,
        z_vars: torch.Tensor,
        cell_type_idx: torch.Tensor
    ):
        """
        Fit empirical prior from scRNA-seq latent representations.
        
        Args:
            z_means: Posterior means [n_cells, n_latent]
            z_vars: Posterior variances [n_cells, n_latent]
            cell_type_idx: Cell type indices [n_cells]
        """
        device = z_means.device
        
        for c in range(self.n_cell_types):
            mask = (cell_type_idx == c)
            n_cells_c = mask.sum().item()
            
            if n_cells_c < 2:
                logger.warning(f"Cell type {c} has < 2 cells, using standard normal prior")
                continue
            
            z_c = z_means[mask]
            var_c = z_vars[mask]
            
            if self.prior_type == 'gaussian':
                # Fit single Gaussian
                self.mu[c] = z_c.mean(dim=0)
                # Use empirical variance + average posterior variance
                self.var[c] = z_c.var(dim=0) + var_c.mean(dim=0)
                self.var[c] = self.var[c].clamp(min=self.eps)
            else:
                # Fit GMM using k-means initialization
                self._fit_gmm_single_type(z_c, var_c, c)
        
        self._fitted = True
        logger.info("Empirical prior fitted successfully")
    
    def _fit_gmm_single_type(
        self,
        z: torch.Tensor,
        var: torch.Tensor,
        cell_type: int,
        n_iter: int = 50
    ):
        """Fit GMM for a single cell type using EM."""
        n_cells = z.shape[0]
        n_comp = min(self.n_components, n_cells)
        
        # K-means initialization
        indices = torch.randperm(n_cells)[:n_comp]
        mu = z[indices].clone()
        
        # Initialize with empirical variance
        sigma_sq = var.mean(dim=0).unsqueeze(0).expand(n_comp, -1).clone()
        weights = torch.ones(n_comp, device=z.device) / n_comp
        
        for _ in range(n_iter):
            # E-step: compute responsibilities
            log_resp = self._compute_log_responsibilities(z, mu, sigma_sq, weights)
            resp = F.softmax(log_resp, dim=1)
            
            # M-step: update parameters
            n_k = resp.sum(dim=0) + 1e-6
            weights = n_k / n_cells
            
            for k in range(n_comp):
                mu[k] = (resp[:, k:k+1] * z).sum(dim=0) / n_k[k]
                diff = z - mu[k]
                sigma_sq[k] = (resp[:, k:k+1] * diff.pow(2)).sum(dim=0) / n_k[k]
                sigma_sq[k] = sigma_sq[k].clamp(min=self.eps)
        
        self.mu[cell_type, :n_comp] = mu
        self.var[cell_type, :n_comp] = sigma_sq
        self.weights[cell_type, :n_comp] = weights
    
    def _compute_log_responsibilities(
        self,
        z: torch.Tensor,
        mu: torch.Tensor,
        var: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """Compute log responsibilities for GMM."""
        n_cells, n_latent = z.shape
        n_comp = mu.shape[0]
        
        log_resp = torch.zeros(n_cells, n_comp, device=z.device)
        
        for k in range(n_comp):
            diff = z - mu[k]
            log_prob = -0.5 * torch.sum(
                torch.log(2 * np.pi * var[k]) + diff.pow(2) / var[k],
                dim=-1
            )
            log_resp[:, k] = torch.log(weights[k] + 1e-10) + log_prob
        
        return log_resp
    
    def log_prob(
        self,
        z: torch.Tensor,
        cell_type_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log probability under the empirical prior.
        
        Args:
            z: Latent samples [batch, n_latent]
            cell_type_idx: Cell type indices [batch]
            
        Returns:
            Log probability [batch]
        """
        if not self._fitted:
            # Fall back to standard normal
            return -0.5 * torch.sum(z.pow(2), dim=-1)
        
        batch_size = z.shape[0]
        log_probs = torch.zeros(batch_size, device=z.device)
        
        if self.prior_type == 'gaussian':
            # Gather per-sample parameters
            mu = self.mu[cell_type_idx]
            var = self.var[cell_type_idx]
            
            # Gaussian log probability
            log_probs = -0.5 * torch.sum(
                torch.log(2 * np.pi * var) + (z - mu).pow(2) / var,
                dim=-1
            )
        else:  # gmm
            for i in range(batch_size):
                c = cell_type_idx[i].item()
                log_probs[i] = self._gmm_log_prob_single(z[i], c)
        
        return log_probs
    
    def _gmm_log_prob_single(self, z: torch.Tensor, cell_type: int) -> torch.Tensor:
        """Compute GMM log probability for a single sample."""
        mu = self.mu[cell_type]
        var = self.var[cell_type]
        weights = self.weights[cell_type]
        
        log_probs = []
        for k in range(self.n_components):
            diff = z - mu[k]
            log_prob = -0.5 * torch.sum(
                torch.log(2 * np.pi * var[k]) + diff.pow(2) / var[k]
            )
            log_probs.append(torch.log(weights[k] + 1e-10) + log_prob)
        
        return torch.logsumexp(torch.stack(log_probs), dim=0)
    
    def neg_log_prob(
        self,
        z: torch.Tensor,
        cell_type_idx: torch.Tensor
    ) -> torch.Tensor:
        """Negative log probability (for loss computation)."""
        return -self.log_prob(z, cell_type_idx)
    
    def sample(
        self,
        cell_type_idx: torch.Tensor,
        n_samples: int = 1
    ) -> torch.Tensor:
        """Sample from the prior."""
        batch_size = cell_type_idx.shape[0]
        device = cell_type_idx.device
        
        if self.prior_type == 'gaussian':
            mu = self.mu[cell_type_idx]
            std = torch.sqrt(self.var[cell_type_idx])
            
            if n_samples == 1:
                return mu + torch.randn_like(mu) * std
            else:
                samples = mu.unsqueeze(0) + torch.randn(
                    n_samples, batch_size, self.n_latent, device=device
                ) * std.unsqueeze(0)
                return samples
        else:
            # Sample component then sample from Gaussian
            samples = torch.zeros(n_samples, batch_size, self.n_latent, device=device)
            for i in range(batch_size):
                c = cell_type_idx[i].item()
                comp = torch.multinomial(self.weights[c], n_samples, replacement=True)
                for s in range(n_samples):
                    k = comp[s].item()
                    samples[s, i] = self.mu[c, k] + torch.randn(
                        self.n_latent, device=device
                    ) * torch.sqrt(self.var[c, k])
            
            return samples.squeeze(0) if n_samples == 1 else samples

# ==============================================================================
# SECTION 6.5: PSEUDO-SPOT GENERATOR (New Section)
# ==============================================================================

class PseudoSpotGenerator:
    """
    Generates synthetic spatial spots by summing scRNA-seq profiles.
    Used for consistency regularization during training.
    """
    def __init__(
        self,
        X_sc: torch.Tensor,
        cell_types: torch.Tensor,
        n_cell_types: int,
        n_pseudo_spots: int = 1000,
        cells_per_spot_range: Tuple[int, int] = (5, 15),
        device: torch.device = DEVICE
    ):
        self.X_sc = X_sc
        self.cell_types = cell_types
        self.n_cell_types = n_cell_types
        self.n_pseudo_spots = n_pseudo_spots
        self.min_cells = cells_per_spot_range[0]
        self.max_cells = cells_per_spot_range[1]
        self.device = device
        
        # Pre-compute indices for each cell type for fast sampling
        self.type_indices = defaultdict(list)
        for idx, ct in enumerate(cell_types.cpu().numpy()):
            self.type_indices[ct].append(idx)
        for k, v in self.type_indices.items():
            self.type_indices[k] = torch.tensor(v, device=device)
            
        self.n_cells_total = X_sc.shape[0]

    def generate_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a batch of pseudo-spots.
        
        Returns:
            X_pseudo: [batch_size, n_genes] (Summed expression)
            proportions: [batch_size, n_cell_types] (Known ground truth)
        """
        X_pseudo_list = []
        props_list = []
        
        # We process generation in chunks for efficiency
        for _ in range(batch_size):
            # 1. Determine how many cells in this spot
            n_cells = np.random.randint(self.min_cells, self.max_cells + 1)
            
            # 2. Randomly sample cell indices
            # Simple random sampling (does not enforce specific composition)
            # This reflects "natural" random co-localization
            indices = torch.randint(0, self.n_cells_total, (n_cells,), device=self.device)
            
            # 3. Sum expression
            # shape: [n_genes]
            spot_expr = self.X_sc[indices].sum(dim=0)
            
            # 4. Calculate proportions
            spot_ct = self.cell_types[indices]
            props = torch.zeros(self.n_cell_types, device=self.device)
            
            # Count occurrences
            unique, counts = torch.unique(spot_ct, return_counts=True)
            props[unique.long()] = counts.float()
            
            # Normalize
            props = props / props.sum()
            
            X_pseudo_list.append(spot_expr)
            props_list.append(props)
            
        return torch.stack(X_pseudo_list), torch.stack(props_list)

# ==============================================================================
# SECTION 6: stVAE (STVAE) - SPATIAL TRANSCRIPTOMICS MODEL
# ==============================================================================

class GammaAmortizer(nn.Module):
    """
    Amortized inference network for spatial γ values.
    
    Maps spatial expression to cell-type-specific latent states:
    x_s -> {γ_sc}_c for all cell types c
    """
    
    def __init__(self, config: STVAEConfig):
        super().__init__()
        
        self.config = config
        self.n_cell_types = config.n_cell_types
        self.n_latent = config.n_latent
        
        # Shared encoder
        self.shared_encoder = MLP(
            in_features=config.n_genes,
            out_features=config.n_hidden,
            hidden_features=config.n_hidden,
            n_layers=config.n_layers_encoder - 1,
            use_batch_norm=config.use_batch_norm,
            use_layer_norm=config.use_layer_norm,
            activation=config.activation,
            dropout_rate=config.dropout_rate,
            output_activation=config.activation
        )
        
        # Cell-type-specific heads
        self.gamma_heads = nn.ModuleList([
            nn.Linear(config.n_hidden, config.n_latent)
            for _ in range(config.n_cell_types)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Spatial expression [batch, n_genes]
            
        Returns:
            gamma: Cell-type-specific latents [batch, n_cell_types, n_latent]
        """
        # Log normalize
        log_x = torch.log1p(x)
        
        # Shared encoding
        h = self.shared_encoder(log_x)
        
        # Cell-type-specific gamma
        gammas = []
        for c in range(self.n_cell_types):
            gamma_c = self.gamma_heads[c](h)
            gammas.append(gamma_c)
        
        # Stack: [batch, n_cell_types, n_latent]
        gamma = torch.stack(gammas, dim=1)
        
        return gamma

class SpotProportions(nn.Module):
    """
    Module for cell type proportions at each spatial spot.
    
    Learns π_s such that Σ_c π_sc = 1.
    Optionally includes an "unknown" cell type category.
    """
    
    def __init__(
        self,
        n_spots: int,
        n_cell_types: int,
        add_unknown: bool = True,
        init_scale: float = 0.1,  # INCREASED from 0.01
        temperature: float = 0.5   # ADD temperature parameter
    ):
        super().__init__()
        
        self.n_spots = n_spots
        self.n_cell_types = n_cell_types
        self.add_unknown = add_unknown
        self.temperature = temperature  # Lower = sharper distributions
        
        n_categories = n_cell_types + 1 if add_unknown else n_cell_types
        
        # Learnable logits - larger init_scale allows more differentiation
        self.logits = nn.Embedding(n_spots, n_categories)
        nn.init.normal_(self.logits.weight, mean=0.0, std=init_scale)
        
        # If unknown is added, initialize it with negative bias to discourage it
        if add_unknown:
            with torch.no_grad():
                self.logits.weight[:, -1] = -2.0  # Penalize unknown initially
    
    def forward(self, spot_idx: torch.Tensor) -> torch.Tensor:
        """Get raw logits for given spots."""
        return self.logits(spot_idx)
    
    def get_proportions(
        self,
        spot_idx: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None
    ) -> torch.Tensor:
        """
        Get softmax-normalized proportions.
        
        Args:
            spot_idx: Spot indices, or None for all spots
            temperature: Softmax temperature (lower = sharper)
            
        Returns:
            Proportions [n_spots, n_categories] or [batch, n_categories]
        """
        if spot_idx is None:
            spot_idx = torch.arange(self.n_spots, device=self.logits.weight.device)
        
        if temperature is None:
            temperature = self.temperature
        
        logits = self.forward(spot_idx)
        return F.softmax(logits / temperature, dim=-1)
    
    def get_known_proportions(
        self,
        spot_idx: Optional[torch.Tensor] = None,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Get proportions for known cell types only.
        
        Args:
            spot_idx: Spot indices
            normalize: Whether to renormalize known proportions
            
        Returns:
            Proportions [batch, n_cell_types]
        """
        props = self.get_proportions(spot_idx)
        known_props = props[:, :self.n_cell_types]
        
        if normalize and self.add_unknown:
            known_props = known_props / (known_props.sum(dim=-1, keepdim=True) + 1e-8)
        
        return known_props

class stVAE(nn.Module):
    """
    Spatial Transcriptomics Latent Variable Model (STVAE).
    
    Deconvolution model that:
    1. Estimates cell type proportions π_s at each spot
    2. Infers cell-type-specific states γ_sc at each spot
    3. Uses pre-trained scVAE decoder for expression profiles
    
    Generative model:
        x_sg | π_s, {γ_sc} ~ NB(l_s * α_g * Σ_c π_sc * ρ_g(γ_sc, c), θ_g)
    
    where:
        - π_s: Cell type proportions at spot s
        - γ_sc: Cell-type-specific state at spot s for type c
        - α_g: Gene-specific correction factor
        - ρ: Decoder from scVAE
    """
    
    def __init__(
        self,
        sc_model: scVAE,
        config: STVAEConfig,
        freeze_sc_decoder: bool = True,
        n_spots: Optional[int] = None  # ADD THIS PARAMETER
    ):
        super().__init__()
        
        self.config = config
        self.n_genes = config.n_genes
        self.n_cell_types = config.n_cell_types
        self.n_latent = config.n_latent
        
        # Reuse scVAE decoder (frozen)
        self.sc_decoder = sc_model.decoder
        self.log_theta_sc = sc_model.log_theta
        
        if freeze_sc_decoder:
            for param in self.sc_decoder.parameters():
                param.requires_grad = False
            self.log_theta_sc.requires_grad = False
        
        # Gamma inference network - MODIFIED LOGIC
        if config.amortized_gamma:
            self.gamma_network = GammaAmortizer(config)
            self.gamma_free = None
        else:
            self.gamma_network = None
            # Use n_spots from parameter or config
            actual_n_spots = n_spots if n_spots is not None else config.n_spots
            if actual_n_spots <= 0:
                raise ValueError(
                    "n_spots must be specified for non-amortized gamma. "
                    "Either pass n_spots parameter or set config.n_spots."
                )
            self.gamma_free = nn.Parameter(
                torch.randn(actual_n_spots, config.n_cell_types, config.n_latent) * 0.1
            )
            logger.info(f"Initialized non-amortized gamma with shape: {self.gamma_free.shape}")
        
        # Gene-specific correction factor α
        self.log_alpha = nn.Parameter(torch.zeros(config.n_genes))
        
        # Optional: learnable dispersion adjustment for spatial data
        self.log_theta_adjustment = nn.Parameter(torch.zeros(config.n_genes))
    
    @property
    def alpha(self) -> torch.Tensor:
        """Gene correction factor, constrained to [alpha_min, alpha_max]."""
        alpha = torch.exp(self.log_alpha)
        alpha = alpha.clamp(min=self.config.alpha_min, max=self.config.alpha_max)
        return alpha
    
    @property
    def theta(self) -> torch.Tensor:
        """Spatial dispersion parameter."""
        log_theta = self.log_theta_sc + self.log_theta_adjustment
        return torch.exp(log_theta).clamp(min=1e-4, max=1e4)
    
    def infer_gamma(
        self,
        x: torch.Tensor,
        spot_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Infer cell-type-specific γ values.
        
        Args:
            x: Spatial expression [batch, n_genes]
            spot_idx: Spot indices (required for non-amortized)
            
        Returns:
            gamma: [batch, n_cell_types, n_latent]
        """
        if self.gamma_network is not None:
            return self.gamma_network(x)
        else:
            if spot_idx is None:
                raise ValueError("spot_idx required for non-amortized gamma")
            return self.gamma_free[spot_idx]
    
    def decode_cell_type(
        self,
        gamma: torch.Tensor,
        cell_type: int
    ) -> torch.Tensor:
        """
        Decode expression profile for a specific cell type.
        
        Args:
            gamma: Latent states [batch, n_latent]
            cell_type: Cell type index
            
        Returns:
            rho: Normalized expression profile [batch, n_genes]
        """
        batch_size = gamma.shape[0]
        ct_idx = torch.full(
            (batch_size,), cell_type,
            device=gamma.device, dtype=torch.long
        )
        return self.sc_decoder(gamma, ct_idx)
    
    def decode_all_cell_types(
        self,
        gamma: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode expression profiles for all cell types.
        
        Args:
            gamma: Latent states [batch, n_cell_types, n_latent]
            
        Returns:
            rho: Expression profiles [batch, n_cell_types, n_genes]
        """
        batch_size = gamma.shape[0]
        device = gamma.device
        
        rho_list = []
        for c in range(self.n_cell_types):
            ct_idx = torch.full((batch_size,), c, device=device, dtype=torch.long)
            rho_c = self.sc_decoder(gamma[:, c, :], ct_idx)
            rho_list.append(rho_c)
        
        return torch.stack(rho_list, dim=1)
    
    def forward(
        self,
        x: torch.Tensor,
        spot_idx: Optional[torch.Tensor], # Can be None if using known_props
        proportions_module: Optional[SpotProportions], 
        library_size: Optional[torch.Tensor] = None,
        known_proportions: Optional[torch.Tensor] = None # NEW ARGUMENT
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for spatial deconvolution.
        
        Args:
            x: Spatial expression [batch, n_genes]
            spot_idx: Spot indices [batch]
            proportions: SpotProportions module
            library_size: Library sizes [batch, 1]
            
        Returns:
            Dictionary with model outputs
        """
        if library_size is None:
            library_size = x.sum(dim=-1, keepdim=True)
        
        # --- LOGIC BRANCHING ---
        if known_proportions is not None:
            # PSEUDO-SPOT MODE: Use Ground Truth Proportions
            pi = known_proportions
            pi_known = pi[:, :self.n_cell_types]
            
            # For pseudo-spots, we MUST use the amortized gamma network
            # because there are no free parameters for synthetic spots
            if self.gamma_network is None:
                raise ValueError("Must use amortized_gamma=True to use pseudo-spots.")
            gamma = self.gamma_network(x)
            
        else:
            # REAL SPOT MODE: Use Learned Proportions
            pi = proportions_module.get_proportions(spot_idx)
            pi_known = pi[:, :self.n_cell_types]
            gamma = self.infer_gamma(x, spot_idx)

        # --- REST IS SAME ---
        rho = self.decode_all_cell_types(gamma)
        
        pi_expanded = pi_known.unsqueeze(-1)
        rho_mixed = (pi_expanded * rho).sum(dim=1)
        
        alpha = self.alpha.unsqueeze(0)
        mu = library_size * alpha * rho_mixed
        
        return {
            'mu': mu,
            'theta': self.theta,
            'pi': pi,
            'pi_known': pi_known,
            'gamma': gamma,
            'rho': rho,
            'rho_mixed': rho_mixed,
            'alpha': self.alpha
        }
    
# ==============================================================================
# SECTION 6.7: PROPORTION-ONLY MODELS
# ==============================================================================
# Location: After class `stVAE` (Section 6), before `class STVAELoss` (Section 7)

class ReferenceProfiles(nn.Module):
    """
    Computes and stores fixed reference expression profiles from scRNA-seq.
    Similar to RCTD's approach of using mean expression per cell type.
    """
    
    def __init__(
        self,
        X_sc: torch.Tensor,
        cell_types: torch.Tensor,
        n_cell_types: int,
        normalize: str = 'cpm'  # 'cpm', 'log_cpm', or 'raw'
    ):
        super().__init__()
        
        self.n_cell_types = n_cell_types
        self.normalize = normalize
        
        # Compute mean profiles per cell type
        profiles = torch.zeros(n_cell_types, X_sc.shape[1])
        
        for c in range(n_cell_types):
            mask = (cell_types == c)
            if mask.sum() > 0:
                ct_expr = X_sc[mask].float()
                
                # Normalize each cell to CPM first
                if normalize in ['cpm', 'log_cpm']:
                    ct_expr = ct_expr / (ct_expr.sum(dim=1, keepdim=True) + 1e-8) * 1e6
                
                # Take mean across cells
                mean_profile = ct_expr.mean(dim=0)
                
                # Log transform if requested
                if normalize == 'log_cpm':
                    mean_profile = torch.log1p(mean_profile)
                
                profiles[c] = mean_profile
        
        # Normalize profiles to sum to 1 (proportion of reads)
        profiles = profiles / (profiles.sum(dim=1, keepdim=True) + 1e-8)
        
        # Register as buffer (not trainable)
        self.register_buffer('profiles', profiles)
    
    def forward(self) -> torch.Tensor:
        """Return reference profiles [n_cell_types, n_genes]."""
        return self.profiles


class NNLSDeconvolution:
    """
    Non-Negative Least Squares deconvolution (similar to RCTD/MuSiC).
    Not a neural network - uses scipy's nnls solver.
    """
    
    def __init__(self, reference_profiles: np.ndarray):
        """
        Args:
            reference_profiles: [n_cell_types, n_genes] normalized profiles
        """
        self.profiles = reference_profiles  # [C, G]
    
    def deconvolve(self, X_spatial: np.ndarray) -> np.ndarray:
        """
        Deconvolve spatial expression using NNLS.
        
        Args:
            X_spatial: [n_spots, n_genes] spatial expression
            
        Returns:
            proportions: [n_spots, n_cell_types] estimated proportions
        """
        from scipy.optimize import nnls
        
        n_spots = X_spatial.shape[0]
        n_cell_types = self.profiles.shape[0]
        proportions = np.zeros((n_spots, n_cell_types))
        
        # Normalize spatial data to CPM
        X_norm = X_spatial / (X_spatial.sum(axis=1, keepdims=True) + 1e-8) * 1e6
        
        # Solve NNLS for each spot
        # Model: X_s ≈ Σ_c π_sc * profile_c
        # This is: X_s ≈ profiles.T @ π_s
        # NNLS solves: min ||A @ x - b||^2 s.t. x >= 0
        
        A = self.profiles.T  # [G, C]
        
        for s in range(n_spots):
            b = X_norm[s]  # [G]
            x, _ = nnls(A, b)
            proportions[s] = x
        
        # Normalize to sum to 1
        proportions = proportions / (proportions.sum(axis=1, keepdims=True) + 1e-8)
        
        return proportions


class SoftmaxRegressionDeconvolution(nn.Module):
    """
    Learnable deconvolution with softmax-constrained proportions.
    Uses fixed reference profiles but learns spot-specific proportions.
    """
    
    def __init__(
        self,
        n_spots: int,
        n_cell_types: int,
        reference_profiles: torch.Tensor,
        temperature: float = 0.5,
        add_unknown: bool = False
    ):
        super().__init__()
        
        self.n_spots = n_spots
        self.n_cell_types = n_cell_types
        self.temperature = temperature
        self.add_unknown = add_unknown
        
        # Fixed reference profiles [C, G]
        self.register_buffer('profiles', reference_profiles)
        
        n_categories = n_cell_types + 1 if add_unknown else n_cell_types
        
        # Learnable proportion logits [n_spots, n_categories]
        self.logits = nn.Embedding(n_spots, n_categories)
        nn.init.normal_(self.logits.weight, mean=0.0, std=0.1)
        
        # Gene-specific correction factor (like Cell2location's sensitivity)
        self.log_alpha = nn.Parameter(torch.zeros(reference_profiles.shape[1]))
    
    @property
    def alpha(self) -> torch.Tensor:
        return torch.exp(self.log_alpha).clamp(min=0.1, max=10.0)
    
    def get_proportions(
        self,
        spot_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get softmax-normalized proportions."""
        if spot_idx is None:
            spot_idx = torch.arange(self.n_spots, device=self.logits.weight.device)
        
        logits = self.logits(spot_idx)
        return F.softmax(logits / self.temperature, dim=-1)
    
    def forward(
        self,
        spot_idx: torch.Tensor,
        library_size: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute expected expression using fixed profiles.
        
        mu_sg = l_s * α_g * Σ_c π_sc * ρ_cg
        """
        # Get proportions [batch, C] or [batch, C+1]
        pi = self.get_proportions(spot_idx)
        
        # Use only known cell types for reconstruction
        pi_known = pi[:, :self.n_cell_types]
        
        # Mix profiles: [batch, G] = [batch, C] @ [C, G]
        rho_mixed = torch.matmul(pi_known, self.profiles)
        
        # Scale by library size and alpha
        mu = library_size * self.alpha.unsqueeze(0) * rho_mixed
        
        return {
            'mu': mu,
            'pi': pi,
            'pi_known': pi_known,
            'rho_mixed': rho_mixed,
            'alpha': self.alpha
        }


class stAE(nn.Module):
    """
    Autoencoder that maps spatial expression to simplex (proportions).
    The bottleneck is constrained to be a valid probability distribution.
    
    This is lighter than full VAE but allows some learned transformation.
    """
    
    def __init__(
        self,
        n_genes: int,
        n_cell_types: int,
        reference_profiles: torch.Tensor,
        hidden_dim: int = 128,
        temperature: float = 0.5,
        add_unknown: bool = False
    ):
        super().__init__()
        
        self.n_genes = n_genes
        self.n_cell_types = n_cell_types
        self.temperature = temperature
        self.add_unknown = add_unknown
        
        n_output = n_cell_types + 1 if add_unknown else n_cell_types
        
        # Fixed reference profiles
        self.register_buffer('profiles', reference_profiles)
        
        # Encoder: X -> proportions
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_output)
        )
        
        # Gene correction factor
        self.log_alpha = nn.Parameter(torch.zeros(n_genes))
    
    @property
    def alpha(self) -> torch.Tensor:
        return torch.exp(self.log_alpha).clamp(min=0.1, max=10.0)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode expression to proportion logits."""
        log_x = torch.log1p(x)
        logits = self.encoder(log_x)
        return logits
    
    def get_proportions(self, x: torch.Tensor) -> torch.Tensor:
        """Get softmax-normalized proportions."""
        logits = self.encode(x)
        return F.softmax(logits / self.temperature, dim=-1)
    
    def forward(
        self,
        x: torch.Tensor,
        library_size: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        if library_size is None:
            library_size = x.sum(dim=-1, keepdim=True)
        
        pi = self.get_proportions(x)
        pi_known = pi[:, :self.n_cell_types]
        
        # Reconstruct using fixed profiles
        rho_mixed = torch.matmul(pi_known, self.profiles)
        mu = library_size * self.alpha.unsqueeze(0) * rho_mixed
        
        return {
            'mu': mu,
            'pi': pi,
            'pi_known': pi_known,
            'rho_mixed': rho_mixed,
            'alpha': self.alpha
        }

class ProportionOnlyLoss(nn.Module):
    """
    Loss function for proportion-only models.
    Simpler than full STVAE loss - no gamma penalty.
    """
    
    def __init__(
        self,
        config: STVAEConfig,
        gene_likelihood: str = 'nb'
    ):
        super().__init__()
        self.config = config
        self.gene_likelihood = gene_likelihood
        
        # Learnable dispersion
        self.log_theta = nn.Parameter(torch.zeros(config.n_genes))
    
    @property
    def theta(self) -> torch.Tensor:
        return torch.exp(self.log_theta).clamp(min=1e-4, max=1e4)
    
    def forward(
        self,
        x: torch.Tensor,
        outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute loss."""
        
        # NLL
        if self.gene_likelihood == 'nb':
            log_prob = DistributionUtils.log_nb_positive(
                x, outputs['mu'], self.theta
            )
        else:  # poisson approximation
            log_prob = torch.distributions.Poisson(outputs['mu'] + 1e-8).log_prob(x)
        
        nll = -log_prob.sum(dim=-1)
        
        # Alpha regularization (encourage alpha ≈ 1)
        alpha_penalty = torch.mean((torch.log(outputs['alpha'])).pow(2))
        
        # Proportion entropy (encourage peaky distributions)
        pi = outputs['pi_known']
        pi_clamped = pi.clamp(min=1e-8)
        entropy = -torch.sum(pi_clamped * torch.log(pi_clamped), dim=-1)
        
        total_loss = (
            nll.mean()
            + self.config.lambda_alpha * alpha_penalty
            + self.config.lambda_pi_entropy * entropy.mean()
        )
        
        return {
            'loss': total_loss,
            'nll': nll.mean(),
            'alpha_penalty': alpha_penalty,
            'entropy': entropy.mean()
        }
    
# ==============================================================================
# SECTION 7: LOSS FUNCTIONS
# ==============================================================================

class STVAELoss(nn.Module):
    """
    Complete loss function for STVAE spatial model.
    
    L = NLL + λ_γ * KL_γ + λ_α * L_α + λ_π * L_π
    
    where:
        - NLL: Negative log likelihood
        - KL_γ: KL divergence for γ against empirical prior
        - L_α: Regularization on gene correction factors
        - L_π: Optional sparsity/entropy regularization on proportions
    """
    
    def __init__(self, empirical_prior, config):
        super().__init__()
        self.prior = empirical_prior
        self.config = config
        self.lambda_gamma = config.lambda_gamma
        self.lambda_alpha = config.lambda_alpha
        self.lambda_pi_entropy = config.lambda_pi_entropy
        self.lambda_pi_sparsity = config.lambda_pi_sparsity

    def forward(
        self,
        x: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        spot_idx: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        
        device = x.device
        
        # NLL calculation
        if self.config.gene_likelihood == 'nb':
            log_prob = DistributionUtils.log_nb_positive(
                x, outputs['mu'], outputs['theta']
            )
        else:
            raise ValueError(f"Unknown likelihood: {self.config.gene_likelihood}")
        nll = -log_prob.sum(dim=-1)

        # Gamma Penalty
        batch_size = x.shape[0]
        gamma = outputs['gamma']
        pi_known = outputs['pi_known']
        gamma_penalty = torch.zeros(batch_size, device=device)
        
        for c in range(gamma.shape[1]):
            gamma_c = gamma[:, c, :]
            ct_idx = torch.full((batch_size,), c, device=device, dtype=torch.long)
            weight_c = pi_known[:, c].detach()
            neg_log_prior = self.prior.neg_log_prob(gamma_c, ct_idx)
            gamma_penalty += weight_c * neg_log_prior

        # Alpha Penalty
        alpha = outputs['alpha']
        alpha_penalty = torch.mean((torch.log(alpha)).pow(2))
        
        # --- FIXED PROPORTION REGULARIZATION ---
        pi = outputs['pi']
        pi_known = outputs['pi_known']
        
        # 1. Entropy penalty on KNOWN proportions (not including unknown)
        # We want LOW entropy = sharp/peaked distributions
        # Entropy H = -sum(p * log(p)), H is HIGH for uniform, LOW for peaked
        # To minimize entropy, we ADD it to the loss
        pi_clamped = pi_known.clamp(min=1e-8, max=1.0 - 1e-8)
        entropy_per_spot = -torch.sum(pi_clamped * torch.log(pi_clamped), dim=-1)
        pi_entropy = entropy_per_spot.mean()
        
        # 2. Penalty on unknown cell type proportion (encourage it to be small)
        if self.config.add_unknown_cell_type:
            unknown_penalty = pi[:, -1].mean() * 10.0  # Strong penalty on unknown
        else:
            unknown_penalty = torch.tensor(0.0, device=device)
        
        # 3. Sparsity: encourage some proportions to be exactly 0
        # L1 penalty, but we actually want Gini-like sparsity
        if self.lambda_pi_sparsity > 0:
            # Gini coefficient penalty (higher Gini = more sparse)
            sorted_pi, _ = torch.sort(pi_known, dim=-1)
            n = pi_known.shape[-1]
            indices = torch.arange(1, n + 1, device=device).float()
            gini = (2 * torch.sum(indices * sorted_pi, dim=-1) / (n * torch.sum(sorted_pi, dim=-1) + 1e-8)) - (n + 1) / n
            # We want HIGH gini (sparse), so we minimize negative gini
            pi_sparsity = -gini.mean()
        else:
            pi_sparsity = torch.tensor(0.0, device=device)
        
        # Combine losses
        per_spot_loss = nll + self.lambda_gamma * gamma_penalty
        
        total_loss = (
            per_spot_loss.mean()
            + self.lambda_alpha * alpha_penalty
            + self.lambda_pi_entropy * pi_entropy  # Minimize entropy (sharper)
            + self.lambda_pi_sparsity * pi_sparsity
            + unknown_penalty
        )
        
        return {
            'loss': total_loss,
            'nll': nll.mean(),
            'gamma_penalty': gamma_penalty.mean(),
            'alpha_penalty': alpha_penalty,
            'pi_entropy': pi_entropy,
            'pi_sparsity': pi_sparsity,
            'unknown_penalty': unknown_penalty,
            'per_spot_loss': per_spot_loss
        }


# ==============================================================================
# SECTION 8: TRAINING UTILITIES
# ==============================================================================

class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
    
    def __call__(self, loss: float) -> bool:
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class CellTypeWeightCalculator:
    """Calculate cell type weights for balanced training."""
    
    @staticmethod
    def compute_weights(
        cell_type_idx: torch.Tensor,
        n_cell_types: int,
        weight_cap: float = 0.05
    ) -> torch.Tensor:
        """
        Compute inverse-frequency weights for cell types.
        
        Args:
            cell_type_idx: Cell type indices [n_cells]
            n_cell_types: Total number of cell types
            weight_cap: Minimum proportion for rare types
            
        Returns:
            Per-sample weights [n_cells]
        """
        n_cells = len(cell_type_idx)
        
        # Count cells per type
        counts = torch.zeros(n_cell_types)
        for c in range(n_cell_types):
            counts[c] = (cell_type_idx == c).sum()
        
        # Compute proportions with cap
        proportions = counts / n_cells
        proportions = proportions.clamp(min=weight_cap)
        
        # Inverse-frequency weights
        weights = 1.0 / (proportions * n_cell_types)
        
        # Per-sample weights
        sample_weights = weights[cell_type_idx]
        
        # Normalize
        sample_weights = sample_weights / sample_weights.mean()
        
        return sample_weights


# ==============================================================================
# SECTION 9: TRAINERS
# ==============================================================================

class scVAETrainer:
    """
    Trainer for scVAE (Stage 1: Reference Training).
    """
    def __init__(
        self,
        model: scVAE,
        config: STVAEConfig,
        device: torch.device = DEVICE
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.sc_learning_rate, 
            weight_decay=config.sc_weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        self.early_stopping = EarlyStopping(
            patience=config.sc_early_stopping_patience
        ) if config.sc_early_stopping else None

    def train_epoch(
        self, 
        dataloader: DataLoader, 
        sample_weights: torch.Tensor,
        kl_weight: float
    ) -> Dict[str, float]:
        self.model.train()
        epoch_metrics = defaultdict(float)
        n_samples = 0
        
        for batch in dataloader:
            x = batch[0].to(self.device)
            cell_type_idx = batch[1].to(self.device)
            idx = batch[2]
            
            # Get weights for this batch
            weights = sample_weights[idx].to(self.device) if sample_weights is not None else None
            batch_size = x.shape[0]
            
            # Forward and Loss
            loss_dict = self.model.loss(
                x, cell_type_idx, 
                kl_weight=kl_weight, 
                sample_weights=weights
            )
            
            # Optimize
            self.optimizer.zero_grad()
            loss_dict['loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
            self.optimizer.step()
            
            # Track metrics
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor) and value.dim() == 0:
                    epoch_metrics[key] += value.item() * batch_size
            n_samples += batch_size
            
        for key in epoch_metrics:
            epoch_metrics[key] /= n_samples
            
        return dict(epoch_metrics)
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        val_metrics = defaultdict(float)
        n_samples = 0
        
        for batch in dataloader:
            x = batch[0].to(self.device)
            cell_type_idx = batch[1].to(self.device)
            batch_size = x.shape[0]
            
            # Validation loss (KL weight usually 1.0 or same as train)
            loss_dict = self.model.loss(x, cell_type_idx, kl_weight=1.0)
            
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor) and value.dim() == 0:
                    val_metrics[key] += value.item() * batch_size
            n_samples += batch_size
        
        for key in val_metrics:
            val_metrics[key] /= n_samples
            
        return dict(val_metrics)
    
    def fit(
        self,
        X: torch.Tensor,
        cell_types: torch.Tensor,
        n_epochs: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        
        if n_epochs is None:
            n_epochs = self.config.sc_max_epochs
        
        # 1. Compute Class Weights
        if self.config.use_cell_type_weights:
            sample_weights = CellTypeWeightCalculator.compute_weights(
                cell_types,
                self.config.n_cell_types,
                self.config.cell_type_weight_cap
            )
        else:
            sample_weights = torch.ones(len(X))
        
        # 2. Dataloader
        indices = torch.arange(len(X))
        dataset = TensorDataset(X.float(), cell_types.long(), indices)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.sc_batch_size,
            shuffle=True,
            drop_last=True
        )
        
        history = defaultdict(list)
        
        # 3. Training Loop
        for epoch in range(n_epochs):
            # KL Annealing
            kl_weight = min(1.0, epoch / 50.0) * self.config.sc_kl_weight
            
            metrics = self.train_epoch(dataloader, sample_weights, kl_weight)
            
            # Scheduler step
            self.scheduler.step(metrics['loss'])
            
            # History
            for k, v in metrics.items():
                history[k].append(v)
            
            # Logging
            if verbose and (epoch + 1) % 10 == 0:
                lr = self.optimizer.param_groups[0]['lr']
                logger.info(
                    f"[scVAE] Epoch {epoch+1:3d}/{n_epochs} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"KL: {metrics['kl_loss']:.4f} | "
                    f"LR: {lr:.2e}"
                )
                
            # Early Stopping
            if self.early_stopping and self.early_stopping(metrics['loss']):
                if verbose: logger.info(f"Early stopping at epoch {epoch+1}")
                break
                
        return dict(history)

class stVAETrainer:
    """
    Trainer for stVAE (Stage 2: Spatial Deconvolution).
    """
    def __init__(
        self,
        model: stVAE,
        proportions: SpotProportions,
        loss_fn: STVAELoss,
        config: STVAEConfig,
        pseudo_generator: Optional[PseudoSpotGenerator] = None, 
        device: torch.device = DEVICE
    ):
        self.model = model.to(device)
        self.proportions = proportions.to(device)
        self.loss_fn = loss_fn.to(device)
        self.config = config
        self.pseudo_generator = pseudo_generator
        self.device = device
        
        # Optimizer with parameter groups
        param_groups = []
        
        # 1. Gamma params - MODIFIED FOR BOTH MODES
        gamma_params = []
        if model.gamma_network is not None:
            # Amortized mode: encoder network parameters
            gamma_params.extend(model.gamma_network.parameters())
            gamma_lr = config.st_learning_rate
        else:
            # Non-amortized mode: per-spot gamma parameters
            gamma_params.append(model.gamma_free)
            # Use higher learning rate for direct optimization
            gamma_lr = config.st_learning_rate * 2.0
        
        param_groups.append({
            'params': gamma_params, 
            'lr': gamma_lr, 
            'name': 'gamma'
        })
        
        # 2. Alpha/Theta params
        param_groups.append({
            'params': [model.log_alpha, model.log_theta_adjustment], 
            'lr': config.st_learning_rate * 0.1, 
            'name': 'alpha'
        })
        
        # 3. Proportion params (Higher LR often helps here)
        param_groups.append({
            'params': list(proportions.parameters()), 
            'lr': config.st_learning_rate * 5.0, 
            'name': 'proportions'
        })
        
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=config.st_weight_decay)
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=100, T_mult=2, eta_min=1e-6
        )
        
        self.early_stopping = EarlyStopping(
            patience=config.st_early_stopping_patience
        ) if config.st_early_stopping else None
    
    def train_epoch(self, dataloader: DataLoader, epoch: int = 0) -> Dict[str, float]:
        self.model.train()
        self.proportions.train()
        
        epoch_metrics = defaultdict(float)
        n_samples = 0
        
        # Pseudo-spot weight warmup
        w_pseudo = 0.0
        if self.pseudo_generator:
            w_pseudo = self.config.pseudo_weight * \
                       min(1.0, epoch / max(1, self.config.pseudo_warmup_epochs))
        
        for batch in dataloader:
            # --- Real Data ---
            x_real = batch[0].to(self.device)
            spot_idx = batch[1].to(self.device)
            batch_size = x_real.shape[0] # e.g., 128
            
            outputs_real = self.model(x_real, spot_idx, self.proportions)
            loss_dict = self.loss_fn(x_real, outputs_real, spot_idx)
            loss_total = loss_dict['loss']
            
            # --- Pseudo Data (Consistency) ---
            if self.pseudo_generator is not None:
                # MODIFIED LOGIC HERE:
                # Calculate pseudo batch size based on ratio
                # e.g., if ratio is 2, generate 256 pseudo spots for 128 real spots
                n_pseudo_batch = int(batch_size * self.config.pseudo_training_ratio)
                
                x_pseudo, pi_true = self.pseudo_generator.generate_batch(n_pseudo_batch)
                
                # Pass known proportions, spot_idx is None
                outputs_pseudo = self.model(
                    x=x_pseudo, 
                    spot_idx=None, 
                    proportions_module=None, 
                    known_proportions=pi_true
                )
                
                # Simple NLL loss for pseudo spots (Supervised)
                log_prob_p = DistributionUtils.log_nb_positive(
                    x_pseudo, outputs_pseudo['mu'], outputs_pseudo['theta']
                )
                
                # Note: .mean() ensures the loss scale is independent of batch size,
                # so simply adding them works regardless of the ratio.
                nll_pseudo = -log_prob_p.sum(dim=-1).mean()
                
                loss_total += w_pseudo * nll_pseudo
                epoch_metrics['pseudo_nll'] += nll_pseudo.item() * batch_size

            # Optimize
            self.optimizer.zero_grad()
            loss_total.backward()
            
            # Clip gradients
            all_params = list(self.model.parameters()) + list(self.proportions.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, 10.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Metrics
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor) and v.dim() == 0:
                    epoch_metrics[k] += v.item() * batch_size
            n_samples += batch_size
            
        for k in epoch_metrics:
            epoch_metrics[k] /= n_samples
            
        return dict(epoch_metrics)
    
    def fit(
        self,
        X: torch.Tensor,
        n_epochs: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        
        if n_epochs is None:
            n_epochs = self.config.st_max_epochs
        
        n_spots = X.shape[0]
        indices = torch.arange(n_spots)
        dataset = TensorDataset(X.float(), indices)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.st_batch_size,
            shuffle=True
        )
        
        history = defaultdict(list)
        
        for epoch in range(n_epochs):
            metrics = self.train_epoch(dataloader, epoch=epoch)
            
            for k, v in metrics.items():
                history[k].append(v)
            
            if verbose and (epoch + 1) % 50 == 0:
                lr = self.optimizer.param_groups[0]['lr']
                p_str = f"| P-NLL: {metrics.get('pseudo_nll',0):.2f}" if self.pseudo_generator else ""
                logger.info(
                    f"[stVAE] Epoch {epoch+1:4d} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"NLL: {metrics['nll']:.4f} {p_str} | "
                    f"LR: {lr:.2e}"
                )
                
            if self.early_stopping and self.early_stopping(metrics['nll']):
                if verbose: logger.info(f"Early stopping at epoch {epoch+1}")
                break
                
        return dict(history)
    
class stVAETrainerWarmStart(stVAETrainer):
    """Extended trainer that handles loss annealing for training."""
    
    def train_epoch(self, dataloader: DataLoader, epoch: int = 0) -> Dict[str, float]:
        """Train one epoch with annealing update."""
        # Call parent's train_epoch
        metrics = super().train_epoch(dataloader, epoch)
        
        # Update annealing after each epoch
        if hasattr(self.loss_fn, 'step_epoch'):
            self.loss_fn.step_epoch()
            # Optionally log the current lambda value
            if hasattr(self.loss_fn, 'current_lambda'):
                metrics['lambda_constraint'] = self.loss_fn.current_lambda
        
        return metrics
    
# ==============================================================================
# SECTION 9.5: PROPORTION-ONLY TRAINER
# ==============================================================================
# Location: After class `stVAETrainer`, before class `STVAE`

class ProportionOnlyTrainer:
    """
    Trainer for proportion-only deconvolution models.
    """
    
    def __init__(
        self,
        model: Union[SoftmaxRegressionDeconvolution, stAE],
        loss_fn: ProportionOnlyLoss,
        config: STVAEConfig,
        device: torch.device = DEVICE
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.config = config
        self.device = device
        
        # Combine parameters
        params = list(model.parameters()) + list(loss_fn.parameters())
        
        self.optimizer = torch.optim.AdamW(
            params,
            lr=config.st_learning_rate,
            weight_decay=config.st_weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.st_max_epochs,
            eta_min=1e-6
        )
        
        self.early_stopping = EarlyStopping(
            patience=config.st_early_stopping_patience
        ) if config.st_early_stopping else None
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        is_amortized: bool = True
    ) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()
        epoch_metrics = defaultdict(float)
        n_samples = 0
        
        for batch in dataloader:
            x = batch[0].to(self.device)
            spot_idx = batch[1].to(self.device) if len(batch) > 1 else None
            batch_size = x.shape[0]
            
            library_size = x.sum(dim=-1, keepdim=True)
            
            # Forward
            if is_amortized:
                # stAE
                outputs = self.model(x, library_size)
            else:
                # SoftmaxRegressionDeconvolution
                outputs = self.model(spot_idx, library_size)
            
            # Loss
            loss_dict = self.loss_fn(x, outputs)
            
            # Optimize
            self.optimizer.zero_grad()
            loss_dict['loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
            self.optimizer.step()
            
            # Track metrics
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor) and v.dim() == 0:
                    epoch_metrics[k] += v.item() * batch_size
            n_samples += batch_size
        
        self.scheduler.step()
        
        for k in epoch_metrics:
            epoch_metrics[k] /= n_samples
        
        return dict(epoch_metrics)
    
    def fit(
        self,
        X: torch.Tensor,
        n_epochs: Optional[int] = None,
        is_amortized: bool = True,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """Fit the model."""
        if n_epochs is None:
            n_epochs = self.config.st_max_epochs
        
        n_spots = X.shape[0]
        indices = torch.arange(n_spots)
        dataset = TensorDataset(X.float(), indices)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.st_batch_size,
            shuffle=True
        )
        
        history = defaultdict(list)
        
        for epoch in range(n_epochs):
            metrics = self.train_epoch(dataloader, is_amortized)
            
            for k, v in metrics.items():
                history[k].append(v)
            
            if verbose and (epoch + 1) % 100 == 0:
                logger.info(
                    f"[ProportionOnly] Epoch {epoch+1:4d} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"NLL: {metrics['nll']:.4f}"
                )
            
            if self.early_stopping and self.early_stopping(metrics['nll']):
                if verbose:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        return dict(history)

# ==============================================================================
# SECTION 9.7: TRAINER (SIMPLIFIED - FIXED MODE ONLY)
# ==============================================================================

@dataclass
class WarmStartTargets:
    """Container for all AE-derived initialization targets."""
    proportions: np.ndarray           # [n_spots, n_types] - π per spot
    intensity_maps: np.ndarray        # [n_spots, n_types] - cell intensity per spot
    cell_type_names: List[str]        # Cell type names


class WarmStartTrainer:
    """
    Warm-start training with AE-derived proportions and intensity maps.
    Uses FIXED constraints (no annealing).
    """
    
    @staticmethod
    def get_initial_proportions_and_intensity(
        X_st: Union[np.ndarray, torch.Tensor],
        X_sc: Union[np.ndarray, torch.Tensor],
        cell_types: Union[np.ndarray, torch.Tensor],
        n_components: int = 10,
        n_epochs: int = 500,
        config: Optional[STVAEConfig] = None,
        device: torch.device = DEVICE
    ) -> Dict[str, Any]:
        """
        Get initial proportion AND intensity estimates using stAE.
        
        Returns:
            Dict containing proportions, intensity_maps, alpha, and cell_type_names
        """
        logger.info("Computing initial proportions and intensity maps using stAE...")
        
        # Convert to numpy if needed
        if isinstance(X_st, torch.Tensor):
            X_st_np = X_st.cpu().numpy()
        else:
            X_st_np = X_st
            
        if isinstance(X_sc, torch.Tensor):
            X_sc_np = X_sc.cpu().numpy()
        else:
            X_sc_np = X_sc
            
        if isinstance(cell_types, torch.Tensor):
            cell_types_np = cell_types.cpu().numpy()
        else:
            cell_types_np = cell_types
        
        # Get unique cell types
        unique_types = np.unique(cell_types_np)
        n_cell_types = len(unique_types)
        cell_type_names = [str(ct) for ct in unique_types]
        
        # Convert to tensors for model
        X_sc_tensor = torch.from_numpy(X_sc_np).float().to(device)
        X_st_tensor = torch.from_numpy(X_st_np).float().to(device)
        cell_types_tensor = torch.from_numpy(cell_types_np).long().to(device)
        
        # Compute reference profiles
        ref_profiles = ReferenceProfiles(
            X_sc_tensor.cpu(), cell_types_tensor.cpu(), n_cell_types, normalize='cpm'
        )
        profiles_tensor = ref_profiles.profiles.to(device)
        
        n_spots = X_st_np.shape[0]
        n_genes = X_st_np.shape[1]
        library_sizes = X_st_tensor.sum(dim=-1, keepdim=True)
        
        if config is None:
            config = STVAEConfig(n_genes=n_genes, n_cell_types=n_cell_types)
        
        model = stAE(
            n_genes=n_genes,
            n_cell_types=n_cell_types,
            reference_profiles=profiles_tensor,
            hidden_dim=config.n_hidden if hasattr(config, 'n_hidden') else 128,
            temperature=config.proportion_temperature if hasattr(config, 'proportion_temperature') else 0.5,
            add_unknown=False
        ).to(device)
        
        loss_fn = ProportionOnlyLoss(config)
        trainer = ProportionOnlyTrainer(model, loss_fn, config, device)
        trainer.fit(X_st_tensor, n_epochs=n_epochs, is_amortized=True, verbose=False)
        
        model.eval()
        with torch.no_grad():
            outputs = model(X_st_tensor, library_sizes)
            proportions = outputs['pi_known'].cpu().numpy()
            alpha = outputs['alpha'].cpu().numpy()  # NEW: Extract alpha
            
            # Compute intensity maps
            profile_sums = (alpha[np.newaxis, :] * profiles_tensor.cpu().numpy()).sum(axis=-1)
            intensity_maps = library_sizes.cpu().numpy() * proportions * profile_sums[np.newaxis, :]
            
            # Normalize intensity maps
            intensity_max = intensity_maps.max(axis=0, keepdims=True)
            intensity_max[intensity_max == 0] = 1.0
            intensity_normalized = intensity_maps / intensity_max
        
        return {
            'proportions': proportions,
            'intensity_maps': intensity_normalized,
            'intensity_maps_raw': intensity_maps,  # NEW: Also return raw intensity
            'alpha': alpha,  # NEW: Return alpha
            'cell_type_names': cell_type_names,
            'reference_profiles': profiles_tensor.cpu().numpy()  # NEW: Return profiles
        }


class FixedProportionIntensityLoss(nn.Module):
    """
    Loss function that enforces fixed constraints from stAE initialization.
    
    This loss wraps the base STVAELoss and adds constraint enforcement:
    
    1. freeze_proportions=True:
       - Replaces learned π with fixed stAE proportions
       - Ensures exact matching of proportion estimates
       - Only γ (cell states) contributes to reconstruction
    
    2. freeze_intensity=True:
       - Adds MSE penalty between predicted and target intensity patterns
       - Does NOT directly freeze intensity (which is derived, not learned)
       - Instead, constrains the model to maintain stAE intensity patterns
       - Combined with freezing α (requires_grad=False in model init)
    
    Note on α (gene correction) vs Intensity:
    ----------------------------------------
    - α (alpha): Learnable gene-specific correction factor [n_genes]
      Captures platform/technical differences per gene
      
    - Intensity: DERIVED value, not directly parameterized
      intensity_c = library_size × π_c × Σ_g(α_g × profile_cg)
      
    When --freeze_intensity is set:
      1. α.requires_grad = False (in stVAE.__init__)
      2. This loss adds intensity pattern matching penalty
      
    This ensures both the gene correction AND the resulting intensity
    patterns remain close to stAE estimates.
    
    Parameters:
    ----------
    base_loss : STVAELoss
        The underlying loss function with gamma prior penalties
    fixed_proportions : torch.Tensor, optional
        Target proportions from stAE [n_spots, n_cell_types]
    fixed_intensity : torch.Tensor, optional
        Target intensity maps from stAE [n_spots, n_cell_types]
    freeze_proportions : bool
        Whether to enforce exact proportion matching
    freeze_intensity : bool
        Whether to penalize deviation from target intensity patterns
    intensity_loss_weight : float
        Weight for intensity matching penalty (default: 10.0)
    """
    
    def __init__(
        self,
        base_loss: STVAELoss,
        fixed_proportions: Optional[torch.Tensor] = None,
        fixed_intensity: Optional[torch.Tensor] = None,
        freeze_proportions: bool = True,
        freeze_intensity: bool = False,
        intensity_loss_weight: float = 10.0
    ):
        super().__init__()
        self.base_loss = base_loss
        self.freeze_proportions = freeze_proportions
        self.freeze_intensity = freeze_intensity
        self.intensity_loss_weight = intensity_loss_weight
        
        if fixed_proportions is not None:
            self.register_buffer('fixed_proportions', fixed_proportions)
        else:
            self.fixed_proportions = None
            
        if fixed_intensity is not None:
            self.register_buffer('fixed_intensity', fixed_intensity)
        else:
            self.fixed_intensity = None
    
    def forward(
        self,
        x: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        spot_idx: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute loss with fixed constraints."""
        
        # Override proportions if fixing
        if self.freeze_proportions and self.fixed_proportions is not None:
            outputs = dict(outputs)  # Make a copy
            outputs['pi_known'] = self.fixed_proportions[spot_idx]
            outputs['pi'] = self.fixed_proportions[spot_idx]
        
        # Get base loss
        base_loss_dict = self.base_loss(x, outputs, spot_idx)
        
        total_loss = base_loss_dict['loss']
        intensity_penalty = torch.tensor(0.0, device=x.device)
        
        # Add intensity constraint if fixing
        if self.freeze_intensity and self.fixed_intensity is not None:
            library_size = x.sum(dim=-1, keepdim=True)
            pi_known = outputs['pi_known']
            
            # Predicted intensity per cell type
            predicted_intensity = library_size * pi_known
            target_intensity = self.fixed_intensity[spot_idx]
            
            # Normalize both to [0, 1] range per spot
            pred_norm = predicted_intensity / (predicted_intensity.sum(dim=-1, keepdim=True) + 1e-8)
            target_norm = target_intensity / (target_intensity.sum(dim=-1, keepdim=True) + 1e-8)
            
            intensity_penalty = F.mse_loss(pred_norm, target_norm)
            total_loss = total_loss + self.intensity_loss_weight * intensity_penalty
        
        return {
            'loss': total_loss,
            'base_loss': base_loss_dict['loss'],
            'intensity_penalty': intensity_penalty,
            'nll': base_loss_dict.get('nll', torch.tensor(0.0)),
            'gamma_penalty': base_loss_dict.get('gamma_penalty', torch.tensor(0.0))
        }

class SoftConstraintLoss(nn.Module):
    """
    Loss function with SOFT constraints on proportions.
    
    Instead of freezing π to stAE values, allows π to be learned
    but penalizes deviation from stAE prior via KL divergence.
    
    This breaks the γ-π correlation problem by:
    1. Allowing π to adjust slightly to fit data better
    2. Penalizing large deviations from stAE prior
    3. Optionally decorrelating γ from π directly
    
    Parameters:
    ----------
    base_loss : STVAELoss
        The underlying loss function with gamma prior penalties
    pi_prior : torch.Tensor
        Target proportions from stAE [n_spots, n_cell_types]
    constraint_strength : float
        Weight for KL divergence penalty (higher = closer to hard freeze)
    decorrelate_weight : float
        Weight for γ-π decorrelation penalty
    """
    
    def __init__(
        self,
        base_loss: STVAELoss,
        pi_prior: torch.Tensor,
        constraint_strength: float = 10.0,
        decorrelate_weight: float = 0.0
    ):
        super().__init__()
        self.base_loss = base_loss
        self.constraint_strength = constraint_strength
        self.decorrelate_weight = decorrelate_weight
        
        # Register prior as buffer
        self.register_buffer('pi_prior', pi_prior)
    
    def kl_divergence_pi(
        self,
        pi_learned: torch.Tensor,
        spot_idx: torch.Tensor,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        KL divergence between learned π and prior π from stAE.
        KL(learned || prior) = Σ learned * log(learned / prior)
        """
        pi_prior_batch = self.pi_prior[spot_idx]
        
        # Clamp for numerical stability
        pi_learned = pi_learned.clamp(min=eps, max=1.0 - eps)
        pi_prior_batch = pi_prior_batch.clamp(min=eps, max=1.0 - eps)
        
        # KL divergence per spot
        kl = torch.sum(
            pi_learned * (torch.log(pi_learned) - torch.log(pi_prior_batch)),
            dim=-1
        )
        return kl.mean()
    
    def correlation_penalty(
        self,
        gamma: torch.Tensor,
        pi: torch.Tensor,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Penalize correlation between gamma and pi.
        Forces gamma to capture information NOT explained by pi.
        
        Args:
            gamma: [batch, n_cell_types, n_latent]
            pi: [batch, n_cell_types]
        """
        batch_size = gamma.shape[0]
        n_cell_types = gamma.shape[1]
        n_latent = gamma.shape[2]
        
        total_corr = torch.tensor(0.0, device=gamma.device)
        
        for c in range(n_cell_types):
            # Get gamma for this cell type: [batch, n_latent]
            gamma_c = gamma[:, c, :]
            # Get pi for this cell type: [batch]
            pi_c = pi[:, c]
            
            # Standardize
            gamma_std = (gamma_c - gamma_c.mean(dim=0, keepdim=True)) / (gamma_c.std(dim=0, keepdim=True) + eps)
            pi_std = (pi_c - pi_c.mean()) / (pi_c.std() + eps)
            
            # Compute correlation for each latent dimension
            # correlation = (1/N) * sum(gamma_std * pi_std)
            corr_per_dim = torch.abs(torch.sum(gamma_std * pi_std.unsqueeze(-1), dim=0) / batch_size)
            
            total_corr = total_corr + corr_per_dim.mean()
        
        return total_corr / n_cell_types
    
    def forward(
        self,
        x: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        spot_idx: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute loss with soft constraints."""
        
        # Get base loss (includes NLL, gamma penalty, etc.)
        base_loss_dict = self.base_loss(x, outputs, spot_idx)
        
        total_loss = base_loss_dict['loss']
        
        # Add soft constraint on proportions (KL divergence)
        pi_known = outputs['pi_known']
        kl_pi = self.kl_divergence_pi(pi_known, spot_idx)
        pi_constraint_loss = self.constraint_strength * kl_pi
        total_loss = total_loss + pi_constraint_loss
        
        # Add decorrelation penalty if requested
        decorr_loss = torch.tensor(0.0, device=x.device)
        if self.decorrelate_weight > 0:
            gamma = outputs['gamma']
            decorr_loss = self.decorrelate_weight * self.correlation_penalty(gamma, pi_known)
            total_loss = total_loss + decorr_loss
        
        return {
            'loss': total_loss,
            'base_loss': base_loss_dict['loss'],
            'kl_pi': kl_pi,
            'pi_constraint_loss': pi_constraint_loss,
            'decorr_loss': decorr_loss,
            'nll': base_loss_dict.get('nll', torch.tensor(0.0)),
            'gamma_penalty': base_loss_dict.get('gamma_penalty', torch.tensor(0.0))
        }
    
# ==============================================================================
# SECTION 10: MAIN DESTVI CLASS
# ==============================================================================

class STVAE:
    """
    Main interface for STVAE spatial deconvolution.
    
    This class provides a high-level API for:
    1. Training scVAE on single-cell reference
    2. Training stVAE on spatial data
    3. Extracting proportions and cell states
    4. Imputing cell-type-specific expression
    
    Example:
        >>> stvae = STVAE(config)
        >>> stvae.fit_sc(X_sc, cell_types)
        >>> stvae.fit_spatial(X_spatial)
        >>> proportions = stvae.get_proportions()
        >>> gamma = stvae.get_cell_type_states(X_spatial)
    """
    
    def __init__(self, config: Optional[STVAEConfig] = None):
        """
        Initialize STVAE.
        
        Args:
            config: Model configuration, uses defaults if None
        """
        self.config = config if config is not None else STVAEConfig()
        self.device = DEVICE
        
        # Models
        self.sc_model: Optional[scVAE] = None
        self.st_model: Optional[stVAE] = None
        self.proportions: Optional[SpotProportions] = None
        self.empirical_prior: Optional[EmpiricalPrior] = None
        self.loss_fn: Optional[STVAELoss] = None
        
        # Training history
        self.sc_history: Dict[str, List[float]] = {}
        self.st_history: Dict[str, List[float]] = {}
        
        # State flags
        self._sc_fitted = False
        self._st_fitted = False
        
        # Store data info
        self._gene_names: Optional[List[str]] = None
        self._cell_type_names: Optional[List[str]] = None
    
    def fit_sc(
        self,
        X: Union[np.ndarray, torch.Tensor],
        cell_types: Union[np.ndarray, torch.Tensor],
        gene_names: Optional[List[str]] = None,
        cell_type_names: Optional[List[str]] = None,
        n_epochs: Optional[int] = None,
        verbose: bool = True
    ) -> 'STVAE':
        """
        Fit scVAE on single-cell reference data.
        
        Args:
            X: Expression matrix [n_cells, n_genes]
            cell_types: Cell type indices [n_cells]
            gene_names: Optional gene names
            cell_type_names: Optional cell type names
            n_epochs: Number of training epochs
            verbose: Print training progress
            
        Returns:
            self for method chaining
        """
        logger.info("=" * 60)
        logger.info("Stage 1: Training scVAE on single-cell reference")
        logger.info("=" * 60)
        
        # Convert to tensors
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(cell_types, np.ndarray):
            cell_types = torch.from_numpy(cell_types).long()
        
        # Update config
        n_cells, n_genes = X.shape
        unique_types = torch.unique(cell_types)
        n_cell_types = len(unique_types)
        
        self.config.n_cells = n_cells
        self.config.n_genes = n_genes
        self.config.n_cell_types = n_cell_types
        
        logger.info(f"Data: {n_cells} cells, {n_genes} genes, {n_cell_types} cell types")
        
        # Store names
        self._gene_names = gene_names
        self._cell_type_names = cell_type_names
        
        # Initialize model
        self.sc_model = scVAE(self.config)
        
        # Train
        trainer = scVAETrainer(self.sc_model, self.config, self.device)
        self.sc_history = trainer.fit(
            X, cell_types,
            n_epochs=n_epochs,
            verbose=verbose
        )
        
        # Fit empirical prior
        logger.info("\nFitting empirical prior from scRNA-seq latents...")
        self._fit_empirical_prior(X, cell_types)
        
        self._sc_fitted = True
        logger.info("scVAE training complete!\n")
        
        return self
    
    def _fit_empirical_prior(
        self,
        X: torch.Tensor,
        cell_types: torch.Tensor
    ):
        """Fit empirical prior from scRNA-seq data."""
        self.sc_model.eval()
        
        with torch.no_grad():
            X_device = X.to(self.device)
            ct_device = cell_types.to(self.device)
            
            z_mu, z_var = self.sc_model.get_latent_representation(
                X_device, ct_device, return_variance=True
            )
        
        # Initialize and fit prior
        self.empirical_prior = EmpiricalPrior(
            n_cell_types=self.config.n_cell_types,
            n_latent=self.config.n_latent,
            prior_type=self.config.prior_type,
            n_components=self.config.n_prior_components
        )
        
        self.empirical_prior.fit(z_mu.cpu(), z_var.cpu(), cell_types)
        self.empirical_prior.to(self.device)
    
    def fit_spatial(
        self,
        X: Union[np.ndarray, torch.Tensor],
        n_epochs: Optional[int] = None,
        pseudo_generator: Optional['PseudoSpotGenerator'] = None, 
        verbose: bool = True
    ) -> 'STVAE':
        """
        Fit stVAE on spatial transcriptomics data.
        """
        if not self._sc_fitted:
            raise RuntimeError("Must call fit_sc() before fit_spatial()")
        
        logger.info("=" * 60)
        logger.info("Stage 2: Training stVAE on spatial data")
        logger.info(f"Inference Mode: {'Amortized' if self.config.amortized_gamma else 'Non-Amortized'}")
        logger.info("=" * 60)
        
        # Convert to tensor
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        
        # Validate dimensions
        n_spots, n_genes = X.shape
        assert n_genes == self.config.n_genes, \
            f"Gene mismatch: spatial has {n_genes}, scRNA-seq has {self.config.n_genes}"
        
        self.config.n_spots = n_spots
        logger.info(f"Data: {n_spots} spots, {n_genes} genes")
        
        # Check pseudo-spot compatibility with non-amortized mode
        if pseudo_generator is not None and not self.config.amortized_gamma:
            logger.warning(
                "Pseudo-spot training is not compatible with non-amortized gamma. "
                "Disabling pseudo-spot training."
            )
            pseudo_generator = None
        
        # Initialize spatial model - PASS n_spots for non-amortized
        self.st_model = stVAE(
            sc_model=self.sc_model,
            config=self.config,
            freeze_sc_decoder=True,
            n_spots=n_spots  # ADD THIS
        )
        
        # Initialize proportions
        self.proportions = SpotProportions(
            n_spots=n_spots,
            n_cell_types=self.config.n_cell_types,
            add_unknown=self.config.add_unknown_cell_type,
            temperature=self.config.proportion_temperature 
        )
        
        # Initialize loss function
        self.loss_fn = STVAELoss(
            empirical_prior=self.empirical_prior,
            config=self.config
        )
        
        # Train
        trainer = stVAETrainer(
            self.st_model,
            self.proportions,
            self.loss_fn,
            self.config,
            pseudo_generator=pseudo_generator,
            device=self.device
        )
        
        self.st_history = trainer.fit(
            X,
            n_epochs=n_epochs,
            verbose=verbose
        )
        
        self._st_fitted = True
        logger.info("stVAE training complete!\n")
        
        return self
    
    def fit_spatial_two_stage(
        self,
        X: np.ndarray,
        X_sc: np.ndarray,
        cell_types: np.ndarray,
        coords: Optional[np.ndarray] = None,
        n_epochs: Optional[int] = None,
        verbose: bool = True,
        freeze_proportions: bool = False,
        freeze_intensity: bool = False,
        soft_constraint: bool = False,           # NEW
        constraint_strength: float = 10.0,       # NEW
        decorrelate_gamma: float = 0.0           # NEW
    ) -> Dict[str, Any]:
        """
        Two-stage training pipeline for spatial deconvolution.
        
        Pipeline Overview:
        -----------------
        Stage 2a (stAE - already run via WarmStartTrainer):
            - Input: X_st, reference profiles from X_sc
            - Learns: π (proportions), α (gene correction)
            - Output: Final proportion and intensity estimates
        
        Stage 2b (stVAE - this method):
            - Input: X_st, frozen scVAE decoder, optional frozen π/α
            - Learns: γ (cell-type-specific latent states)
            - Output: Continuous cell state variation per spot per cell type
        
        Parameters:
        ----------
        X : np.ndarray
            Spatial expression matrix [n_spots, n_genes]
        X_sc : np.ndarray
            Single-cell reference [n_cells, n_genes]
        cell_types : np.ndarray
            Cell type labels for scRNA-seq [n_cells]
        coords : np.ndarray, optional
            Spatial coordinates [n_spots, 2]
        n_epochs : int, optional
            Training epochs for Stage 2b
        verbose : bool
            Print training progress
        freeze_proportions : bool
            If True, fix π to stAE values (only γ is learned)
        freeze_intensity : bool
            If True, fix α (gene correction) to stAE values.
            Note: This freezes the gene-specific correction factor, NOT
            the intensity maps directly. Intensity is derived as:
            intensity = library_size × π × (α · profile).sum()
        
        Returns:
        -------
        Dict containing training history and comparison metrics
        
        Mathematical Model:
        ------------------
        stAE (Stage 2a):
            μ_sg = l_s × α_g × Σ_c π_sc × P̄_cg
            where P̄_cg is the FIXED mean profile from scRNA-seq
        
        stVAE (Stage 2b):
            μ_sg = l_s × α_g × Σ_c π_sc × ρ_g(γ_sc, c)
            where ρ is the FROZEN scVAE decoder output
            
        When freeze_proportions=True:
            - π_sc is FIXED to stAE values
            - Only γ_sc is optimized
            - γ learns to explain expression variation not captured by fixed π
        
        When freeze_intensity=True (freeze α):
            - α_g is FIXED to stAE values
            - Prevents gene correction from changing during γ optimization
            - More constrained: γ must explain ALL remaining variation
        """
        logger.info("=" * 70)
        logger.info("TWO-STAGE TRAINING PIPELINE")
        logger.info("=" * 70)
        logger.info("")
        logger.info("Stage 2a (stAE): Proportion & Intensity Estimation")
        logger.info("  - Uses FIXED reference profiles (mean expression per cell type)")
        logger.info("  - Learns: π (proportions), α (gene correction factor)")
        logger.info("  - Outputs: proportions.csv, intensity_maps.csv (FINAL results)")
        logger.info("")
        logger.info("Stage 2b (stVAE): Cell State Inference")
        logger.info("  - Uses FROZEN scVAE decoder")
        logger.info("  - Learns: γ (cell-type-specific latent states)")
        logger.info("")
        
        if soft_constraint:
            logger.info("CONSTRAINT: Proportions (π) SOFT constrained via KL divergence")
            logger.info(f"  -> Constraint strength: {constraint_strength}")
            logger.info(f"  -> Decorrelation weight: {decorrelate_gamma}")
            logger.info("  -> π can deviate from stAE but is penalized")
            logger.info("  -> This breaks γ-π correlation while maintaining good proportions")
        elif freeze_proportions:
            logger.info("CONSTRAINT: Proportions (π) HARD FROZEN to stAE values")
            logger.info("  -> Only γ (cell states) will be optimized")
            logger.info("  -> WARNING: May cause γ-π correlation artifacts")
        
        if freeze_intensity:
            logger.info("CONSTRAINT: Gene correction (α) FROZEN")
            logger.info("  -> α captures gene-specific technical effects")
            logger.info("  -> Intensity = lib_size × π × (α · profile).sum()")
            logger.info("  -> Both α and derived intensity patterns are preserved")
        
        if not freeze_proportions and not freeze_intensity and not soft_constraint:
            logger.warning("No constraints set. Consider using fit_spatial() for unconstrained training.")
        
        # === STAGE 1: Get initial proportions AND intensity maps from AE ===
        logger.info("\n--- Stage 1: Computing initial proportions and intensity maps ---")
        
        two_stage_results = WarmStartTrainer.get_initial_proportions_and_intensity(
            X_st=X,
            X_sc=X_sc,
            cell_types=cell_types,
            n_components=self.config.n_latent,
            device=self.device
        )
        
        initial_proportions = two_stage_results['proportions']
        initial_intensity = two_stage_results['intensity_maps']
        cell_type_names = two_stage_results['cell_type_names']
        
        logger.info(f"Initial proportions shape: {initial_proportions.shape}")
        logger.info(f"Initial intensity shape: {initial_intensity.shape}")
        logger.info(f"Cell types: {cell_type_names}")
        
        # Log initial proportion statistics
        logger.info("\nInitial proportion statistics (from AE):")
        for i, ct in enumerate(cell_type_names):
            props = initial_proportions[:, i]
            logger.info(f"  {ct}: mean={props.mean():.4f}, std={props.std():.4f}, "
                    f"min={props.min():.4f}, max={props.max():.4f}")
        
        # === STAGE 2: Fit scVAE if not already fitted ===
        if not self._sc_fitted:
            logger.info("\n--- Stage 2a: Fitting scVAE reference model ---")
            self.fit_sc(
                X=X_sc,
                cell_types=cell_types,
                cell_type_names=cell_type_names,
                verbose=verbose
            )
        
        # === STAGE 3: Initialize ST model ===
        logger.info("\n--- Stage 2b: Initializing ST model ---")
        
        n_spots, n_genes = X.shape
        n_cell_types = len(cell_type_names)
        self.config.n_spots = n_spots
        
        # Convert to tensor
        if isinstance(X, np.ndarray):
            X_tensor = torch.from_numpy(X).float()
        else:
            X_tensor = X.float()
        
        # Initialize ST model
        self.st_model = stVAE(
            sc_model=self.sc_model,
            config=self.config,
            freeze_sc_decoder=True,
            n_spots=n_spots
        )
        
        # Freeze alpha/theta if freeze_intensity is True
        if freeze_intensity:
            self.st_model.log_alpha.requires_grad = False
            self.st_model.log_theta_adjustment.requires_grad = False
            logger.info("Alpha and theta frozen (requires_grad=False) for fixed intensity mode")
        
        # Initialize proportions module with AE values
        self.proportions = SpotProportions(
            n_spots=n_spots,
            n_cell_types=n_cell_types,
            add_unknown=self.config.add_unknown_cell_type,
            temperature=self.config.proportion_temperature
        )
        
        # Initialize proportion logits from AE proportions
        with torch.no_grad():
            eps = 1e-6
            props_clamped = np.clip(initial_proportions, eps, 1 - eps)
            logits = np.log(props_clamped)
            logits = logits - logits.mean(axis=1, keepdims=True)
            
            if self.config.add_unknown_cell_type:
                unknown_logits = np.full((n_spots, 1), -2.0)
                logits = np.concatenate([logits, unknown_logits], axis=1)
            
            self.proportions.logits.weight.data = torch.tensor(
                logits, dtype=torch.float32, device=self.device
            )
            logger.info("Initialized proportion logits from AE")
        
        # === STAGE 4: Setup loss function ===
        logger.info("\n--- Stage 3: Setting up loss function ---")
        
        # Convert targets to tensors
        target_proportions = torch.tensor(
            initial_proportions, dtype=torch.float32, device=self.device
        )
        target_intensity = torch.tensor(
            initial_intensity, dtype=torch.float32, device=self.device
        )
        
        # Create base loss
        base_loss = STVAELoss(
            empirical_prior=self.empirical_prior,
            config=self.config
        )
        
        # === MODIFIED: Choose between hard freeze and soft constraint ===
        if soft_constraint:
            # Soft constraint mode: allow π to be learned but regularized
            logger.info(f"Using SOFT constraint on proportions (strength={constraint_strength})")
            logger.info(f"Decorrelation weight: {decorrelate_gamma}")
            
            self.loss_fn = SoftConstraintLoss(
                base_loss=base_loss,
                pi_prior=target_proportions,
                constraint_strength=constraint_strength,
                decorrelate_weight=decorrelate_gamma
            ).to(self.device)
            
            # Do NOT freeze proportions - they should be learnable
            # But we can still freeze intensity if requested
            if freeze_intensity:
                self.st_model.log_alpha.requires_grad = False
                self.st_model.log_theta_adjustment.requires_grad = False
                logger.info("Alpha and theta frozen for fixed intensity mode")
                
        else:
            # Original hard freeze mode
            self.loss_fn = FixedProportionIntensityLoss(
                base_loss=base_loss,
                fixed_proportions=target_proportions if freeze_proportions else None,
                fixed_intensity=target_intensity if freeze_intensity else None,
                freeze_proportions=freeze_proportions,
                freeze_intensity=freeze_intensity,
                intensity_loss_weight=10.0
            ).to(self.device)
            
            # Freeze proportions if fixing (hard mode)
            if freeze_proportions:
                for param in self.proportions.parameters():
                    param.requires_grad = False
                logger.info("Proportions frozen (requires_grad=False) - HARD FREEZE")
            
            # Freeze alpha/theta if freeze_intensity is True
            if freeze_intensity:
                self.st_model.log_alpha.requires_grad = False
                self.st_model.log_theta_adjustment.requires_grad = False
                logger.info("Alpha and theta frozen for fixed intensity mode")
        # === END MODIFICATION ===
        
        total_epochs = n_epochs or self.config.st_max_epochs
        
        # Create trainer
        trainer = stVAETrainerWarmStart(
            model=self.st_model,
            proportions=self.proportions,
            loss_fn=self.loss_fn,
            config=self.config,
            device=self.device
        )
        
        # Freeze proportions if fixing
        if freeze_proportions:
            for param in self.proportions.parameters():
                param.requires_grad = False
            logger.info("Proportions frozen (requires_grad=False)")
        
        logger.info(f"Training for {total_epochs} epochs with FIXED constraints")
        
        self.st_history = trainer.fit(
            X_tensor,
            n_epochs=total_epochs,
            verbose=verbose
        )
        
        # === STAGE 5: Compute final results ===
        logger.info("\n--- Stage 4: Computing final results ---")
        
        self._st_fitted = True
        final_proportions = self.get_proportions()
        
        # Store stVAE-refined proportions separately (for later saving)
        self.stvae_refined_proportions = final_proportions.copy()
        
        # Compare with initial AE proportions
        logger.info("\nComparison: AE vs Final VAE proportions:")
        logger.info("-" * 60)
        
        total_mae = 0.0
        total_corr = 0.0
        
        for i, ct in enumerate(cell_type_names):
            ae_props = initial_proportions[:, i]
            vae_props = final_proportions[:, i]
            
            mae = np.abs(ae_props - vae_props).mean()
            if ae_props.std() > 0:
                corr = np.corrcoef(ae_props, vae_props)[0, 1]
            else:
                corr = 1.0
            
            total_mae += mae
            total_corr += corr
            
            logger.info(f"  {ct}:")
            logger.info(f"    AE:  mean={ae_props.mean():.4f}, std={ae_props.std():.4f}")
            logger.info(f"    VAE: mean={vae_props.mean():.4f}, std={vae_props.std():.4f}")
            logger.info(f"    MAE={mae:.6f}, Correlation={corr:.4f}")
        
        avg_mae = total_mae / n_cell_types
        avg_corr = total_corr / n_cell_types
        
        logger.info("-" * 60)
        logger.info(f"OVERALL: Average MAE={avg_mae:.6f}, Average Correlation={avg_corr:.4f}")
        
        if freeze_proportions:
            logger.info(f"(Expected: MAE≈0, Corr≈1.0 since proportions were FIXED)")
        
        # Store results
        self.two_stage_results = {
            'ae_proportions': initial_proportions,
            'ae_intensity': initial_intensity,
            'vae_proportions': final_proportions,
            'cell_type_names': cell_type_names,
            'avg_mae': avg_mae,
            'avg_corr': avg_corr,
            'freeze_proportions': freeze_proportions,
            'freeze_intensity': freeze_intensity
        }
        
        return self.st_history
    
    def fit(
        self,
        X_sc: Union[np.ndarray, torch.Tensor],
        cell_types: Union[np.ndarray, torch.Tensor],
        X_spatial: Union[np.ndarray, torch.Tensor],
        gene_names: Optional[List[str]] = None,
        cell_type_names: Optional[List[str]] = None,
        sc_epochs: Optional[int] = None,
        st_epochs: Optional[int] = None,
        verbose: bool = True
    ) -> 'STVAE':
        """
        Convenience method to fit both stages.
        
        Args:
            X_sc: Single-cell expression [n_cells, n_genes]
            cell_types: Cell type indices [n_cells]
            X_spatial: Spatial expression [n_spots, n_genes]
            gene_names: Optional gene names
            cell_type_names: Optional cell type names
            sc_epochs: Epochs for scVAE
            st_epochs: Epochs for stVAE
            verbose: Print progress
            
        Returns:
            self
        """
        self.fit_sc(
            X_sc, cell_types,
            gene_names=gene_names,
            cell_type_names=cell_type_names,
            n_epochs=sc_epochs,
            verbose=verbose
        )
        
        self.fit_spatial(
            X_spatial,
            n_epochs=st_epochs,
            verbose=verbose
        )
        
        return self
    
    def fit_spatial_proportion_only(
        self,
        X: Union[np.ndarray, torch.Tensor],
        X_sc: Union[np.ndarray, torch.Tensor],
        cell_types: Union[np.ndarray, torch.Tensor],
        method: str = 'nnls',
        n_epochs: Optional[int] = None,
        verbose: bool = True
    ) -> 'STVAE':
        """
        Fit proportion-only deconvolution (RCTD/Cell2location-style).
        
        Does NOT require scVAE to be fitted first.
        Uses fixed reference profiles instead of learned decoder.
        
        Args:
            X: Spatial expression [n_spots, n_genes]
            X_sc: Single-cell reference [n_cells, n_genes]
            cell_types: Cell type indices [n_cells]
            method: 'nnls', 'softmax_regression', or 'simplex_ae'
            n_epochs: Training epochs (ignored for NNLS)
            verbose: Print progress
            
        Returns:
            self
        """
        logger.info("=" * 60)
        logger.info(f"Proportion-Only Deconvolution (Method: {method})")
        logger.info("=" * 60)
        
        # Convert to tensors/arrays
        if isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()
            X = X.float()
        else:
            X_np = X
            X = torch.from_numpy(X).float()
        
        if isinstance(X_sc, torch.Tensor):
            X_sc_tensor = X_sc.float()
        else:
            X_sc_tensor = torch.from_numpy(X_sc).float()
        
        if isinstance(cell_types, np.ndarray):
            cell_types_tensor = torch.from_numpy(cell_types).long()
        else:
            cell_types_tensor = cell_types.long()
        
        n_spots, n_genes = X.shape
        n_cell_types = len(torch.unique(cell_types_tensor))
        
        self.config.n_spots = n_spots
        self.config.n_genes = n_genes
        self.config.n_cell_types = n_cell_types
        
        logger.info(f"Data: {n_spots} spots, {n_genes} genes, {n_cell_types} cell types")
        
        # Compute reference profiles
        ref_profiles = ReferenceProfiles(
            X_sc_tensor, cell_types_tensor, n_cell_types, normalize='cpm'
        )
        profiles_np = ref_profiles.profiles.numpy()
        profiles_tensor = ref_profiles.profiles.to(self.device)
        
        if method == 'nnls':
            # Non-parametric NNLS (no training)
            logger.info("Running NNLS deconvolution...")
            nnls_model = NNLSDeconvolution(profiles_np)
            self._proportions_np = nnls_model.deconvolve(X_np)
            self._proportion_method = 'nnls'
            
        elif method == 'softmax_regression':
            # Learnable proportions with fixed profiles
            logger.info("Training softmax regression...")
            self._prop_model = SoftmaxRegressionDeconvolution(
                n_spots=n_spots,
                n_cell_types=n_cell_types,
                reference_profiles=profiles_tensor,
                temperature=self.config.proportion_temperature,
                add_unknown=self.config.add_unknown_cell_type
            )
            
            loss_fn = ProportionOnlyLoss(self.config)
            trainer = ProportionOnlyTrainer(
                self._prop_model, loss_fn, self.config, self.device
            )
            self.st_history = trainer.fit(X, n_epochs, is_amortized=False, verbose=verbose)
            self._proportion_method = 'softmax_regression'
            
        elif method == 'simplex_ae':
            # Autoencoder approach
            logger.info("Training simplex autoencoder...")
            self._prop_model = stAE(
                n_genes=n_genes,
                n_cell_types=n_cell_types,
                reference_profiles=profiles_tensor,
                hidden_dim=self.config.n_hidden,
                temperature=self.config.proportion_temperature,
                add_unknown=self.config.add_unknown_cell_type
            )
            
            loss_fn = ProportionOnlyLoss(self.config)
            trainer = ProportionOnlyTrainer(
                self._prop_model, loss_fn, self.config, self.device
            )
            self.st_history = trainer.fit(X, n_epochs, is_amortized=True, verbose=verbose)
            self._proportion_method = 'simplex_ae'
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self._st_fitted = True
        self._X_spatial = X  # Store for get_proportions
        logger.info("Proportion-only deconvolution complete!\n")
        
        return self
    
    @torch.no_grad()
    def get_proportions(
        self,
        include_unknown: bool = False,
        return_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Get estimated cell type proportions for all spots.
        
        Works for both full mode and proportion-only mode.
        """
        if not self._st_fitted:
            raise RuntimeError("Must call fit_spatial() or fit_spatial_proportion_only() first")
        
        # Check if we're in proportion-only mode
        if hasattr(self, '_proportion_method'):
            if self._proportion_method == 'nnls':
                props = self._proportions_np
                if not return_numpy:
                    props = torch.from_numpy(props)
                return props
            
            elif self._proportion_method in ['softmax_regression', 'simplex_ae']:
                self._prop_model.eval()
                
                if self._proportion_method == 'softmax_regression':
                    spot_idx = torch.arange(self.config.n_spots, device=self.device)
                    props = self._prop_model.get_proportions(spot_idx)
                else:  # simplex_ae
                    X_device = self._X_spatial.to(self.device)
                    props = self._prop_model.get_proportions(X_device)
                
                if not include_unknown and self.config.add_unknown_cell_type:
                    props = props[:, :self.config.n_cell_types]
                    props = props / (props.sum(dim=-1, keepdim=True) + 1e-8)
                
                if return_numpy:
                    return props.cpu().numpy()
                return props
        
        # Original full mode logic
        self.proportions.eval()
        
        if include_unknown or not self.config.add_unknown_cell_type:
            props = self.proportions.get_proportions()
        else:
            props = self.proportions.get_known_proportions(normalize=True)
        
        if return_numpy:
            return props.cpu().numpy()
        return props
    
    @torch.no_grad()
    def get_cell_type_states(
        self,
        X_spatial: Union[np.ndarray, torch.Tensor],
        return_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Get cell-type-specific latent states (γ) for each spot.
        
        Args:
            X_spatial: Spatial expression [n_spots, n_genes]
            return_numpy: Return as numpy array
            
        Returns:
            Gamma [n_spots, n_cell_types, n_latent]
        """
        if not self._st_fitted:
            raise RuntimeError("Must call fit_spatial() first")
        
        self.st_model.eval()
        
        if isinstance(X_spatial, np.ndarray):
            X_spatial = torch.from_numpy(X_spatial).float()
        
        X_device = X_spatial.to(self.device)
        n_spots = X_device.shape[0]
        
        # === FIX: Handle non-amortized mode ===
        if self.config.amortized_gamma:
            # Amortized: infer from expression
            gamma = self.st_model.infer_gamma(X_device)
        else:
            # Non-amortized: use stored per-spot parameters
            spot_idx = torch.arange(n_spots, device=self.device)
            gamma = self.st_model.infer_gamma(X_device, spot_idx=spot_idx)
        
        if return_numpy:
            return gamma.cpu().numpy()
        return gamma
    
    @torch.no_grad()
    def impute_expression(
        self,
        X_spatial: Union[np.ndarray, torch.Tensor],
        cell_type: int,
        n_samples: int = 1,
        return_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Impute cell-type-specific expression at each spot.
        
        Args:
            X_spatial: Spatial expression [n_spots, n_genes]
            cell_type: Cell type index to impute
            n_samples: Number of samples for uncertainty
            return_numpy: Return as numpy array
            
        Returns:
            Imputed expression [n_spots, n_genes] or [n_samples, n_spots, n_genes]
        """
        if not self._st_fitted:
            raise RuntimeError("Must call fit_spatial() first")
        
        self.st_model.eval()
        
        if isinstance(X_spatial, np.ndarray):
            X_spatial = torch.from_numpy(X_spatial).float()
        
        X_device = X_spatial.to(self.device)
        n_spots = X_device.shape[0]
        
        # === FIX: Handle non-amortized mode ===
        if self.config.amortized_gamma:
            # Amortized: infer from expression
            gamma = self.st_model.infer_gamma(X_device)
        else:
            # Non-amortized: use stored per-spot parameters
            spot_idx = torch.arange(n_spots, device=self.device)
            gamma = self.st_model.infer_gamma(X_device, spot_idx=spot_idx)
        
        gamma_c = gamma[:, cell_type, :]
        
        # Decode
        ct_idx = torch.full((n_spots,), cell_type, device=self.device, dtype=torch.long)
        rho = self.st_model.sc_decoder(gamma_c, ct_idx)
        
        if return_numpy:
            return rho.cpu().numpy()
        return rho
    
    @torch.no_grad()
    def get_scale_for_cell_type(
        self,
        X_spatial: Union[np.ndarray, torch.Tensor],
        cell_type: int,
        library_size: Optional[Union[np.ndarray, torch.Tensor]] = None,
        return_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Get expression scale (mean) for a specific cell type.
        
        Scale = library_size * alpha * rho(gamma, cell_type)
        
        Args:
            X_spatial: Spatial expression
            cell_type: Cell type index
            library_size: Optional library sizes
            return_numpy: Return as numpy array
            
        Returns:
            Scale [n_spots, n_genes]
        """
        if not self._st_fitted:
            raise RuntimeError("Must call fit_spatial() first")
        
        self.st_model.eval()
        
        if isinstance(X_spatial, np.ndarray):
            X_spatial = torch.from_numpy(X_spatial).float()
        
        X_device = X_spatial.to(self.device)
        n_spots = X_device.shape[0]
        
        if library_size is None:
            library_size = X_device.sum(dim=-1, keepdim=True)
        elif isinstance(library_size, np.ndarray):
            library_size = torch.from_numpy(library_size).float().to(self.device)
            if library_size.dim() == 1:
                library_size = library_size.unsqueeze(-1)
        
        # === FIX: Handle non-amortized mode ===
        if self.config.amortized_gamma:
            gamma = self.st_model.infer_gamma(X_device)
        else:
            spot_idx = torch.arange(n_spots, device=self.device)
            gamma = self.st_model.infer_gamma(X_device, spot_idx=spot_idx)
        
        gamma_c = gamma[:, cell_type, :]
        
        ct_idx = torch.full((n_spots,), cell_type, device=self.device, dtype=torch.long)
        rho = self.st_model.sc_decoder(gamma_c, ct_idx)
        
        # Scale
        alpha = self.st_model.alpha.unsqueeze(0)
        scale = library_size * alpha * rho
        
        if return_numpy:
            return scale.cpu().numpy()
        return scale
    
    @torch.no_grad()
    def get_reconstruction(
        self,
        X_spatial: Union[np.ndarray, torch.Tensor],
        return_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Get reconstructed spatial expression (mixed over cell types).
        
        Args:
            X_spatial: Spatial expression
            return_numpy: Return as numpy array
            
        Returns:
            Reconstructed expression [n_spots, n_genes]
        """
        if not self._st_fitted:
            raise RuntimeError("Must call fit_spatial() first")
        
        self.st_model.eval()
        self.proportions.eval()
        
        if isinstance(X_spatial, np.ndarray):
            X_spatial = torch.from_numpy(X_spatial).float()
        
        X_device = X_spatial.to(self.device)
        n_spots = X_device.shape[0]
        spot_idx = torch.arange(n_spots, device=self.device)
        
        outputs = self.st_model(X_device, spot_idx, self.proportions)
        
        if return_numpy:
            return outputs['mu'].cpu().numpy()
        return outputs['mu']
    
    # ==========================================================================
    # ANALYSIS METHODS
    # ==========================================================================
    
    def differential_expression(
        self,
        X_spatial: Union[np.ndarray, torch.Tensor],
        cell_type: int,
        grouping: Union[np.ndarray, torch.Tensor],
        n_samples: int = 100,
        fdr_threshold: float = 0.05
    ) -> Dict[str, np.ndarray]:
        """
        Differential expression analysis between spot groups for a cell type.
        
        Uses the imputed cell-type-specific expression to compare groups.
        
        Args:
            X_spatial: Spatial expression
            cell_type: Cell type to analyze
            grouping: Binary group labels [n_spots]
            n_samples: Number of posterior samples
            fdr_threshold: FDR threshold for significance
            
        Returns:
            Dictionary with DE results
        """
        if not self._st_fitted:
            raise RuntimeError("Must call fit_spatial() first")
        
        if isinstance(grouping, torch.Tensor):
            grouping = grouping.cpu().numpy()
        
        # Get imputed expression
        imputed = self.impute_expression(X_spatial, cell_type, return_numpy=True)
        
        # Split by groups
        group1 = imputed[grouping == 0]
        group2 = imputed[grouping == 1]
        
        # Compute statistics
        n_genes = imputed.shape[1]
        log_fc = np.zeros(n_genes)
        p_values = np.zeros(n_genes)
        
        from scipy.stats import mannwhitneyu
        
        for g in range(n_genes):
            # Log fold change
            mean1 = group1[:, g].mean() + 1e-8
            mean2 = group2[:, g].mean() + 1e-8
            log_fc[g] = np.log2(mean2 / mean1)
            
            # Mann-Whitney U test
            try:
                _, p_values[g] = mannwhitneyu(
                    group1[:, g], group2[:, g], alternative='two-sided'
                )
            except:
                p_values[g] = 1.0
        
        # FDR correction
        from scipy.stats import false_discovery_control
        try:
            fdr = false_discovery_control(p_values)
        except:
            # Benjamini-Hochberg fallback
            n = len(p_values)
            sorted_idx = np.argsort(p_values)
            fdr = np.zeros(n)
            for i, idx in enumerate(sorted_idx):
                fdr[idx] = p_values[idx] * n / (i + 1)
            fdr = np.minimum.accumulate(fdr[::-1])[::-1]
            fdr = np.clip(fdr, 0, 1)
        
        # Significant genes
        significant = fdr < fdr_threshold
        
        return {
            'log_fc': log_fc,
            'p_values': p_values,
            'fdr': fdr,
            'significant': significant,
            'gene_names': self._gene_names
        }
    
    def spatial_smoothing(
        self,
        values: np.ndarray,
        coords: np.ndarray,
        bandwidth: float = 1.0,
        kernel: str = 'gaussian'
    ) -> np.ndarray:
        """
        Apply spatial smoothing to values.
        
        Args:
            values: Values to smooth [n_spots] or [n_spots, n_features]
            coords: Spatial coordinates [n_spots, 2]
            bandwidth: Smoothing bandwidth
            kernel: Kernel type ('gaussian' or 'uniform')
            
        Returns:
            Smoothed values
        """
        n_spots = len(values)
        values = np.atleast_2d(values.T).T  # Ensure 2D
        
        # Compute distance matrix
        distances = cdist(coords, coords)
        
        # Compute kernel weights
        if kernel == 'gaussian':
            weights = np.exp(-distances**2 / (2 * bandwidth**2))
        else:  # uniform
            weights = (distances <= bandwidth).astype(float)
        
        # Normalize weights
        weights = weights / weights.sum(axis=1, keepdims=True)
        
        # Apply smoothing
        smoothed = weights @ values
        
        return smoothed.squeeze()
    
    # ==========================================================================
    # UTILITY METHODS
    # ==========================================================================
    
    def save(self, path: str):
        """Save model to disk."""
        checkpoint = {
            'config': self.config,
            'sc_model_state': self.sc_model.state_dict() if self.sc_model else None,
            'st_model_state': self.st_model.state_dict() if self.st_model else None,
            'proportions_state': self.proportions.state_dict() if self.proportions else None,
            'empirical_prior': {
                'mu': self.empirical_prior.mu if self.empirical_prior else None,
                'var': self.empirical_prior.var if self.empirical_prior else None,
                '_fitted': self.empirical_prior._fitted if self.empirical_prior else False
            },
            'sc_history': self.sc_history,
            'st_history': self.st_history,
            '_sc_fitted': self._sc_fitted,
            '_st_fitted': self._st_fitted,
            '_gene_names': self._gene_names,
            '_cell_type_names': self._cell_type_names
        }
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'STVAE':
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=DEVICE)
        
        model = cls(checkpoint['config'])
        
        # Restore scVAE
        if checkpoint['sc_model_state'] is not None:
            model.sc_model = scVAE(model.config)
            model.sc_model.load_state_dict(checkpoint['sc_model_state'])
            model.sc_model.to(DEVICE)
        
        # Restore empirical prior
        if checkpoint['empirical_prior']['mu'] is not None:
            model.empirical_prior = EmpiricalPrior(
                n_cell_types=model.config.n_cell_types,
                n_latent=model.config.n_latent,
                prior_type=model.config.prior_type
            )
            model.empirical_prior.mu = checkpoint['empirical_prior']['mu']
            model.empirical_prior.var = checkpoint['empirical_prior']['var']
            model.empirical_prior._fitted = checkpoint['empirical_prior']['_fitted']
            model.empirical_prior.to(DEVICE)
        
        # Restore stVAE
        if checkpoint['st_model_state'] is not None:
            model.st_model = stVAE(model.sc_model, model.config)
            model.st_model.load_state_dict(checkpoint['st_model_state'])
            model.st_model.to(DEVICE)
        
        # Restore proportions
        if checkpoint['proportions_state'] is not None:
            model.proportions = SpotProportions(
                n_spots=model.config.n_spots,
                n_cell_types=model.config.n_cell_types,
                add_unknown=model.config.add_unknown_cell_type
            )
            model.proportions.load_state_dict(checkpoint['proportions_state'])
            model.proportions.to(DEVICE)
        
        # Restore loss function
        if model.empirical_prior is not None:
            model.loss_fn = STVAELoss(model.empirical_prior, model.config)
        
        # Restore state
        model.sc_history = checkpoint['sc_history']
        model.st_history = checkpoint['st_history']
        model._sc_fitted = checkpoint['_sc_fitted']
        model._st_fitted = checkpoint['_st_fitted']
        model._gene_names = checkpoint.get('_gene_names')
        model._cell_type_names = checkpoint.get('_cell_type_names')
        
        logger.info(f"Model loaded from {path}")
        return model
    
    def summary(self) -> str:
        """Get model summary."""
        lines = [
            "=" * 60,
            "STVAE Model Summary",
            "=" * 60,
            f"Device: {self.device}",
            "",
            "Configuration:",
            f"  - Genes: {self.config.n_genes}",
            f"  - Cell types: {self.config.n_cell_types}",
            f"  - Latent dimensions: {self.config.n_latent}",
            f"  - Hidden dimensions: {self.config.n_hidden}",
            f"  - Gene likelihood: {self.config.gene_likelihood}",
            "",
            "Training Status:",
            f"  - scVAE fitted: {self._sc_fitted}",
            f"  - stVAE fitted: {self._st_fitted}",
        ]
        
        if self._sc_fitted:
            lines.append(f"  - Cells in reference: {self.config.n_cells}")
        
        if self._st_fitted:
            lines.append(f"  - Spatial spots: {self.config.n_spots}")
        
        if self.sc_model is not None:
            n_params_sc = sum(p.numel() for p in self.sc_model.parameters())
            lines.append(f"  - scVAE parameters: {n_params_sc:,}")
        
        if self.st_model is not None:
            n_params_st = sum(p.numel() for p in self.st_model.parameters() if p.requires_grad)
            lines.append(f"  - stVAE trainable parameters: {n_params_st:,}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


# ==============================================================================
# SECTION 12: EVALUATION UTILITIES
# ==============================================================================

class Evaluator:
    """Evaluation utilities for deconvolution results."""
    
    @staticmethod
    def pearson_correlation(
        predicted: np.ndarray,
        true: np.ndarray
    ) -> Dict[str, float]:
        """Compute Pearson correlation per cell type and overall."""
        n_cell_types = true.shape[1]
        
        results = {}
        correlations = []
        
        for c in range(n_cell_types):
            corr, _ = pearsonr(predicted[:, c], true[:, c])
            results[f'type_{c}_pearson'] = corr
            correlations.append(corr)
        
        results['mean_pearson'] = np.mean(correlations)
        
        # Overall correlation
        overall_corr, _ = pearsonr(predicted.flatten(), true.flatten())
        results['overall_pearson'] = overall_corr
        
        return results
    
    @staticmethod
    def spearman_correlation(
        predicted: np.ndarray,
        true: np.ndarray
    ) -> Dict[str, float]:
        """Compute Spearman correlation per cell type and overall."""
        n_cell_types = true.shape[1]
        
        results = {}
        correlations = []
        
        for c in range(n_cell_types):
            corr, _ = spearmanr(predicted[:, c], true[:, c])
            results[f'type_{c}_spearman'] = corr
            correlations.append(corr)
        
        results['mean_spearman'] = np.mean(correlations)
        
        overall_corr, _ = spearmanr(predicted.flatten(), true.flatten())
        results['overall_spearman'] = overall_corr
        
        return results
    
    @staticmethod
    def rmse(predicted: np.ndarray, true: np.ndarray) -> Dict[str, float]:
        """Compute RMSE per cell type and overall."""
        n_cell_types = true.shape[1]
        
        results = {}
        rmses = []
        
        for c in range(n_cell_types):
            rmse = np.sqrt(np.mean((predicted[:, c] - true[:, c])**2))
            results[f'type_{c}_rmse'] = rmse
            rmses.append(rmse)
        
        results['mean_rmse'] = np.mean(rmses)
        results['overall_rmse'] = np.sqrt(np.mean((predicted - true)**2))
        
        return results
    
    @staticmethod
    def jsd(predicted: np.ndarray, true: np.ndarray) -> np.ndarray:
        """
        Jensen-Shannon divergence per spot.
        
        Returns:
            JSD values [n_spots]
        """
        from scipy.spatial.distance import jensenshannon
        
        n_spots = predicted.shape[0]
        jsd_values = np.zeros(n_spots)
        
        for s in range(n_spots):
            jsd_values[s] = jensenshannon(predicted[s], true[s])
        
        return jsd_values
    
    @staticmethod
    def full_evaluation(
        predicted: np.ndarray,
        true: np.ndarray
    ) -> Dict[str, Any]:
        """Complete evaluation with multiple metrics."""
        results = {
            'pearson': Evaluator.pearson_correlation(predicted, true),
            'spearman': Evaluator.spearman_correlation(predicted, true),
            'rmse': Evaluator.rmse(predicted, true),
            'jsd_mean': np.mean(Evaluator.jsd(predicted, true)),
            'jsd_std': np.std(Evaluator.jsd(predicted, true))
        }
        
        return results

# ==============================================================================
# SECTION 12.5: VISUALIZATION UTILITIES (NEW)
# ==============================================================================

class VisualizationUtils:
    @staticmethod
    def _create_hexagon_patch(center, radius, **kwargs):
        """Create a single hexagon patch centered at given coordinates."""
        from matplotlib.patches import RegularPolygon
        return RegularPolygon(
            center, 
            numVertices=6, 
            radius=radius,
            orientation=HEXAGON_ORIENTATION,  # Use global constant
            **kwargs
        )
    
    @staticmethod
    def _calculate_hex_radius(coords: np.ndarray, scale_factor: float = 0.6) -> float:
        """
        Auto-calculate hexagon radius from spot spacing.
        Uses nearest-neighbor distance to estimate appropriate size.
        """
        from scipy.spatial import cKDTree
        tree = cKDTree(coords)
        distances, _ = tree.query(coords, k=2)  # k=2: self + nearest neighbor
        nn_distances = distances[:, 1]  # Exclude self-distance
        median_spacing = np.median(nn_distances)
        return median_spacing * scale_factor
    
    @staticmethod
    def plot_cooccurrence(df_props: pd.DataFrame, output_path: str):
        """Generates a correlation heatmap of cell type proportions."""
        plt.figure(figsize=(10, 8))
        # Compute correlation between cell types
        corr = df_props.corr(method='pearson')
        
        sns.heatmap(
            corr, 
            annot=True, 
            cmap='RdBu_r', 
            center=0,
            vmin=-1,      # ADD THIS
            vmax=1,       # ADD THIS
            fmt='.2f', 
            square=True,
            linewidths=.5,
            annot_kws={
                'size': 10,        # Font size of correlation values
                'weight': 'bold',  # Make values bold
            }
        )
        plt.title("Cell Type Co-occurrence (Correlation)")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

    @staticmethod
    def plot_spatial_maps(
        df_props: pd.DataFrame, 
        coords: np.ndarray, 
        output_dir: str,
        coords_full: Optional[np.ndarray] = None,  # NEW: All coordinates for background
        matched_mask: Optional[np.ndarray] = None,   # NEW: Boolean mask for matched spots
        presence_threshold: float = 0.05
    ):
        """
        Generates spatial maps with HEXAGONAL spot markers:
        1. A grid of plots, one per cell type, showing proportion intensity.
        2. A dominant cell type map.
        Note: This visualizes PROPORTIONS (π), not intensity values.
        Intensity = library_size × π × (α · profile).sum() is a derived quantity.
        
        If coords_full and matched_mask are provided, unmatched spots are shown
        as grey background context.
        """
        global HEXAGON_ORIENTATION  # Add this line at the start of the method
        from matplotlib.patches import RegularPolygon
        from matplotlib.collections import PatchCollection
        
        cell_types = df_props.columns
        n_types = len(cell_types)
        
        # Determine coordinate range from FULL coords if available (for consistent axis limits)
        if coords_full is not None:
            coords_for_limits = coords_full
        else:
            coords_for_limits = coords
        
        # Calculate hexagon radius from data
        hex_radius = VisualizationUtils._calculate_hex_radius(coords_for_limits, scale_factor=0.55)
        
        # Calculate axis limits with padding (consistent across all plots)
        x_min = coords_for_limits[:, 0].min() - hex_radius * 2
        x_max = coords_for_limits[:, 0].max() + hex_radius * 2
        y_min = coords_for_limits[:, 1].min() - hex_radius * 2
        y_max = coords_for_limits[:, 1].max() + hex_radius * 2
        
        # =====================================================================
        # Helper function to add background (unmatched) spots
        # =====================================================================
        def add_background_spots(ax, coords_full, matched_mask, hex_radius):
            """Add grey hexagons for unmatched spots as background."""
            if coords_full is None or matched_mask is None:
                return
            
            unmatched_mask = ~matched_mask
            if not unmatched_mask.any():
                return
            
            unmatched_coords = coords_full[unmatched_mask]
            
            patches = []
            for x, y in unmatched_coords:
                hexagon = RegularPolygon(
                    (x, y),
                    numVertices=6,
                    radius=hex_radius,
                    orientation=HEXAGON_ORIENTATION
                )
                patches.append(hexagon)
            
            # Light grey with low alpha for background
            collection = PatchCollection(
                patches, 
                facecolors='lightgrey', 
                edgecolors='none',
                alpha=0.3
            )
            ax.add_collection(collection)
        
        # =====================================================================
        # 1. Grid Plot (Intensity per type) - HEXAGONAL
        # =====================================================================
        cols = 4
        rows = math.ceil(n_types / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
        axes = axes.flatten()
        
        for i, ct in enumerate(cell_types):
            ax = axes[i]
            values = df_props[ct].values
            
            # ADD: Create mask for spots above threshold
            above_threshold = values >= presence_threshold
            below_threshold = ~above_threshold
            n_above = above_threshold.sum()
            n_below = below_threshold.sum()
            
            # Add background spots FIRST (so they're behind)
            add_background_spots(ax, coords_full, matched_mask, hex_radius)
            
            # Normalize values for colormap (only for spots above threshold)
            values_above = values[above_threshold]
            if len(values_above) > 0:
                vmin, vmax = values_above.min(), values_above.max()
            else:
                vmin, vmax = 0, 1
            if vmax - vmin < 1e-8:
                vmax = vmin + 1e-8
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.cm.plasma

            # Create hexagon patches for matched spots
            patches_above = []
            colors_above = []
            patches_below = []
            
            for j, (x, y) in enumerate(coords):
                hexagon = RegularPolygon(
                    (x, y),
                    numVertices=6,
                    radius=hex_radius,
                    orientation=HEXAGON_ORIENTATION
                )
                # MODIFIED: Separate above/below threshold spots
                if above_threshold[j]:
                    patches_above.append(hexagon)
                    colors_above.append(cmap(norm(values[j])))
                else:
                    patches_below.append(hexagon)
            
            # ADD: Plot below-threshold spots in grey first
            if patches_below:
                collection_below = PatchCollection(
                    patches_below, 
                    facecolors='lightgrey', 
                    edgecolors='none',
                    alpha=0.4
                )
                ax.add_collection(collection_below)
            
            # Plot above-threshold spots with color
            if patches_above:
                collection_above = PatchCollection(
                    patches_above, 
                    facecolors=colors_above, 
                    edgecolors='none'
                )
                ax.add_collection(collection_above)
            
            # Set axis limits (use full coordinate range)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect('equal')
            ax.invert_yaxis()  # Match tissue orientation
            ax.axis('off')
            
            # 1. Top Label: Set fontsize and padding
            ax.set_title(f"{ct}\n(n={n_above} above {presence_threshold:.0%})", 
                        fontsize=14, fontweight='bold', pad=10)
            
            # Prepare ScalarMappable
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            
            # Import ticker locally for formatting
            import matplotlib.ticker as ticker
            
            # 2. Colorbar Size: length (shrink) and width (aspect)
            cbar = plt.colorbar(
                sm, 
                ax=ax, 
                fraction=0.046, 
                pad=0.04,
                shrink=0.85,    # Length: Scales bar to 85% of axis height
                aspect=15       # Width: Ratio of long to short dimension (Higher = Thinner)
            )
            
            # 3. Ticks: Interval of 5 (5 bins) and 2 Decimal Places
            # MaxNLocator(nbins=5) attempts to find ~5 nice intervals
            cbar.locator = ticker.MaxNLocator(nbins=5)
            # NOTE: If you meant a strict step of 0.05, change above line to:
            # cbar.locator = ticker.MultipleLocator(0.05)
            
            cbar.formatter = ticker.FormatStrFormatter('%.2f')
            cbar.update_ticks()
            
            # 4. Label Font Size: Right margin labels
            cbar.ax.tick_params(labelsize=10)
            
        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "spatial_proportion_maps.png"), dpi=300)
        plt.close()

        # =====================================================================
        # 2. Dominant Cell Type Map - HEXAGONAL
        # =====================================================================
        dominant_idx = np.argmax(df_props.values, axis=1)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        cmap = plt.colormaps.get_cmap('tab20').resampled(n_types)
        
        # Add background spots FIRST
        add_background_spots(ax, coords_full, matched_mask, hex_radius)
        
        # Create hexagon patches for dominant type map (matched spots)
        patches = []
        colors = []
        for j, (x, y) in enumerate(coords):
            hexagon = RegularPolygon(
                (x, y),
                numVertices=6,
                radius=hex_radius,
                orientation=HEXAGON_ORIENTATION  # Use global constant
            )
            patches.append(hexagon)
            colors.append(cmap(dominant_idx[j] / max(n_types - 1, 1)))
        
        collection = PatchCollection(patches, facecolors=colors, edgecolors='none')
        ax.add_collection(collection)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')
        
        # Create legend
        handles = [
            plt.Line2D([0], [0], marker='H', color='w', 
                      markerfacecolor=cmap(i / max(n_types - 1, 1)),
                      label=cell_types[i], markersize=12, markeredgecolor='none')
            for i in range(n_types)
        ]
        # Add grey background to legend if there are unmatched spots
        if coords_full is not None and matched_mask is not None and (~matched_mask).any():
            handles.append(
                plt.Line2D([0], [0], marker='H', color='w',
                          markerfacecolor='lightgrey', alpha=0.5,
                          label='No count data', markersize=12, markeredgecolor='none')
            )
        
        ax.legend(handles=handles, title="Cell Type", 
                 bbox_to_anchor=(1.02, 1), loc='upper left',
                 frameon=True, fontsize=10)
        
        plt.title("Dominant Cell Type per Spot", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "spatial_dominant_type.png"), dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_latent_projection(
        latent: np.ndarray, 
        df_props: pd.DataFrame,
        output_path: str,
        spot_size: int = 13  # NEW: configurable spot size parameter
    ):
        """
        Projects latent states (Gamma) to UMAP/t-SNE to visualize spot similarity.
        Enforces isometric axes and places legend outside the plot.
        
        Parameters
        ----------
        latent : np.ndarray
            Latent representation matrix (n_spots x n_latent_dims)
        df_props : pd.DataFrame
            Cell type proportions (n_spots x n_cell_types)
        output_path : str
            Path to save the output figure
        spot_size : int, optional
            Size of scatter plot points (default: 10)
        """
        # Reduce dimensionality
        if HAS_UMAP:
            reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42)
            embedding = reducer.fit_transform(latent)
            method_name = "UMAP"
        else:
            reducer = TSNE(n_components=2, random_state=42)
            embedding = reducer.fit_transform(latent)
            method_name = "t-SNE"

        # Determine dominant cell type for coloring
        dominant_ct = df_props.idxmax(axis=1)
        unique_cts = sorted(dominant_ct.unique())
        n_types = len(unique_cts)
        ct_to_int = {ct: i for i, ct in enumerate(unique_cts)}
        colors = dominant_ct.map(ct_to_int).values

        # Create Figure with axes object for better control
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Use discrete colormap normalization
        cmap = plt.cm.tab20
        
        scatter = ax.scatter(
            embedding[:, 0], 
            embedding[:, 1], 
            c=colors, 
            cmap=cmap,
            vmin=0,
            vmax=max(n_types - 1, 1),
            s=spot_size,  # CHANGED: use parameter instead of hardcoded value
            alpha=0.7,
            edgecolors='none'
        )
        
        # =========================================================================
        # FIX: Force equal axis limits with same scale
        # =========================================================================
        # Get current data range
        x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
        y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()
        
        # Calculate the center and maximum range
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        
        # Use the larger range for both axes (with padding)
        max_range = max(x_max - x_min, y_max - y_min) / 2 * 1.1  # 10% padding
        
        # Set symmetric limits around the center
        ax.set_xlim(x_center - max_range, x_center + max_range)
        ax.set_ylim(y_center - max_range, y_center + max_range)
        
        # Ensure the aspect ratio is equal (1:1 scaling)
        ax.set_aspect('equal', adjustable='box')
        # =========================================================================
        
        # Create legend handles with matching colors
        handles = [
            plt.Line2D(
                [0], [0], 
                marker='o', 
                color='w', 
                markerfacecolor=cmap(i / max(n_types - 1, 1)),
                label=ct, 
                markersize=10,
                markeredgecolor='none'
            ) 
            for i, ct in enumerate(unique_cts)
        ]
        
        # Place legend outside on the right margin
        ax.legend(
            handles=handles, 
            title="Dominant Cell Type", 
            bbox_to_anchor=(1.02, 1.0),
            loc='upper left',
            borderaxespad=0.,
            frameon=True,
            fancybox=True,
            shadow=False
        )
        
        ax.set_title(f"{method_name} of Spatial Latent States (Gamma)")
        ax.set_xlabel(f"{method_name} 1")
        ax.set_ylabel(f"{method_name} 2")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # NEW METHOD: CELL-TYPE-SPECIFIC STATE VISUALIZATION

    @staticmethod
    def plot_cell_type_continuum(
        gamma_states: np.ndarray,
        df_props: pd.DataFrame,
        cell_type_idx: int,
        cell_type_name: str,
        coords: Optional[np.ndarray],
        output_dir: str,
        marker_genes: Optional[Dict[str, np.ndarray]] = None,
        coords_full: Optional[np.ndarray] = None,
        matched_mask: Optional[np.ndarray] = None,
        presence_threshold: float = 0.05,
        latent_combination_method: str = 'pca'  # NEW PARAMETER
    ):
        """
        Visualize the continuous state variation within a single cell type.
        
        Args:
            gamma_states: Full gamma tensor [n_spots, n_cell_types, n_latent]
            df_props: Proportions dataframe
            cell_type_idx: Index of cell type to visualize
            cell_type_name: Name for file saving
            coords: Spatial coordinates [n_spots, 2] or None (matched spots only)
            output_dir: Output directory
            marker_genes: Optional dict {gene_name: expression_array} for coloring
            coords_full: All coordinates including unmatched spots
            matched_mask: Boolean mask for which full coords have counts
            presence_threshold: Minimum proportion to consider cell type "present"
            latent_combination_method: How to combine latent dimensions:
                - 'pca': Use first PC of all latents (captures max variance)
                - 'sum': Simple sum of all latent dimensions
                - 'norm': L2 norm (euclidean distance from origin)
                - 'weighted': Variance-weighted combination
                - 'latent0': Use only first latent dimension (original behavior)
        """
        from sklearn.decomposition import PCA
        
        # Extract states for this cell type only
        ct_gamma = gamma_states[:, cell_type_idx, :]  # [n_spots, n_latent]
        ct_props = df_props.iloc[:, cell_type_idx].values  # [n_spots]
        n_latent = ct_gamma.shape[1]
        
        # Filter spots above threshold
        mask = ct_props > presence_threshold
        ct_gamma_filtered = ct_gamma[mask]
        
        if ct_gamma_filtered.shape[0] < 50:
            print(f"  -> Skipping {cell_type_name}: too few spots above threshold ({mask.sum()} spots)")
            return
        
        # =====================================================================
        # NEW: Combine all latent dimensions into a single score
        # =====================================================================
        if latent_combination_method == 'pca':
            # Use PCA to find the direction of maximum variance
            pca = PCA(n_components=1)
            pca.fit(ct_gamma_filtered)
            latent_score = pca.transform(ct_gamma).flatten()
            score_name = "PC1 (all latents)"
            print(f"  -> {cell_type_name}: PCA explained variance = {pca.explained_variance_ratio_[0]:.2%}")
            
        elif latent_combination_method == 'sum':
            # Simple sum of all latent dimensions
            latent_score = ct_gamma.sum(axis=1)
            score_name = "Sum of Latents"
            
        elif latent_combination_method == 'norm':
            # L2 norm - distance from origin in latent space
            latent_score = np.linalg.norm(ct_gamma, axis=1)
            score_name = "Latent Norm"
            
        elif latent_combination_method == 'weighted':
            # Weight by variance of each dimension (more variable = more important)
            variances = ct_gamma_filtered.var(axis=0)
            variances = variances / (variances.sum() + 1e-8)  # Normalize weights
            latent_score = (ct_gamma * variances).sum(axis=1)
            score_name = "Variance-weighted"
            top_weights = np.sort(variances)[::-1][:3]
            print(f"  -> {cell_type_name}: Top 3 latent weights = {top_weights}")
            
        else:  # 'latent0' or fallback
            # Original behavior: first dimension only
            latent_score = ct_gamma[:, 0]
            score_name = "Latent_0"
        
        # Print score range for comparison
        score_range = latent_score.max() - latent_score.min()
        print(f"  -> {cell_type_name}: {score_name} range = [{latent_score.min():.2f}, {latent_score.max():.2f}] (span: {score_range:.2f})")
        # =====================================================================
        
        # Dimensionality reduction on filtered data for UMAP plot
        if HAS_UMAP and ct_gamma_filtered.shape[0] > 30:
            reducer = umap.UMAP(n_neighbors=min(15, ct_gamma_filtered.shape[0]-1), 
                               min_dist=0.1, random_state=42)
            embedding = reducer.fit_transform(ct_gamma_filtered)
            method_name = "UMAP"
        else:
            reducer = PCA(n_components=2)
            embedding = reducer.fit_transform(ct_gamma_filtered)
            method_name = "PCA"
        
        safe_name = str(cell_type_name).replace("/", "_").replace(" ", "_")
        
        # Plot 1: Color by combined latent score (filtered spots only)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Color by combined score
        latent_score_filtered = latent_score[mask]
        sc1 = axes[0].scatter(
            embedding[:, 0], embedding[:, 1],
            c=latent_score_filtered,
            cmap='coolwarm', s=15, alpha=0.7
        )
        axes[0].set_title(f"{cell_type_name}: Colored by {score_name}")
        axes[0].set_xlabel(f"{method_name} 1")
        axes[0].set_ylabel(f"{method_name} 2")
        plt.colorbar(sc1, ax=axes[0], label=score_name)
        
        # Right: Color by proportion (shows confidence)
        sc2 = axes[1].scatter(
            embedding[:, 0], embedding[:, 1],
            c=ct_props[mask],
            cmap='viridis', s=15, alpha=0.7
        )
        axes[1].set_title(f"{cell_type_name}: Colored by Proportion")
        axes[1].set_xlabel(f"{method_name} 1")
        axes[1].set_ylabel(f"{method_name} 2")
        plt.colorbar(sc2, ax=axes[1], label="Cell Type Proportion")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"continuum_{safe_name}.png"), dpi=300)
        plt.close()
        
        # Plot 2: Spatial distribution of latent state (if coords provided) - HEXAGONAL
        if coords is not None:
            from matplotlib.patches import RegularPolygon
            from matplotlib.collections import PatchCollection
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Use full coordinates for hex radius calculation if available
            coords_for_radius = coords_full if coords_full is not None else coords
            hex_radius = VisualizationUtils._calculate_hex_radius(coords_for_radius, scale_factor=0.55)
            
            # Calculate axis limits from full coordinates
            coords_for_limits = coords_full if coords_full is not None else coords
            x_min = coords_for_limits[:, 0].min() - hex_radius * 2
            x_max = coords_for_limits[:, 0].max() + hex_radius * 2
            y_min = coords_for_limits[:, 1].min() - hex_radius * 2
            y_max = coords_for_limits[:, 1].max() + hex_radius * 2
            
            # Add background (unmatched) spots FIRST
            if coords_full is not None and matched_mask is not None:
                unmatched_mask = ~matched_mask
                if unmatched_mask.any():
                    unmatched_coords = coords_full[unmatched_mask]
                    bg_patches = []
                    for x, y in unmatched_coords:
                        hexagon = RegularPolygon(
                            (x, y),
                            numVertices=6,
                            radius=hex_radius,
                            orientation=HEXAGON_ORIENTATION
                        )
                        bg_patches.append(hexagon)
                    
                    bg_collection = PatchCollection(
                        bg_patches, 
                        facecolors='lightgrey', 
                        edgecolors='none',
                        alpha=0.3
                    )
                    ax.add_collection(bg_collection)
            
            # Use combined latent_score for coloring
            vmin, vmax = latent_score.min(), latent_score.max()
            if vmax - vmin < 1e-8:
                vmax = vmin + 1e-8
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.cm.coolwarm

            # Create mask for spots above/below threshold
            above_threshold = ct_props >= presence_threshold
            
            # Create hexagon patches - separate above and below threshold
            patches_above = []
            colors_above = []
            patches_below = []
            
            for j, (x, y) in enumerate(coords):
                hexagon = RegularPolygon(
                    (x, y),
                    numVertices=6,
                    radius=hex_radius,
                    orientation=HEXAGON_ORIENTATION
                )
                
                if above_threshold[j]:
                    patches_above.append(hexagon)
                    # Color with alpha based on proportion
                    rgba = list(cmap(norm(latent_score[j])))  # USE latent_score
                    rgba[3] = np.clip(ct_props[j] * 3, 0.3, 1.0)
                    colors_above.append(rgba)
                else:
                    patches_below.append(hexagon)
            
            # Plot below-threshold spots in grey FIRST (background)
            if patches_below:
                collection_below = PatchCollection(
                    patches_below, 
                    facecolors='lightgrey', 
                    edgecolors='none',
                    alpha=0.4
                )
                ax.add_collection(collection_below)
            
            # Plot above-threshold spots with color ON TOP
            if patches_above:
                collection_above = PatchCollection(
                    patches_above, 
                    facecolors=colors_above, 
                    edgecolors='none'
                )
                ax.add_collection(collection_above)
            
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.axis('off')
            
            # Colorbar with updated label
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label=f"{cell_type_name} State ({score_name})", shrink=0.8)
            
            # Title with range info
            n_above = above_threshold.sum()
            n_below = (~above_threshold).sum()
            plt.title(
                f"Spatial Distribution of {cell_type_name} State\n"
                f"({score_name}: [{vmin:.1f}, {vmax:.1f}], n={n_above} above {presence_threshold:.0%} threshold)", 
                fontsize=14, fontweight='bold'
            )
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"spatial_state_{safe_name}.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"  -> Saved continuum_{safe_name}.png" + 
              (f" and spatial_state_{safe_name}.png" if coords is not None else ""))

    @staticmethod
    def plot_cell_type_continuum_full(
        gamma_states: np.ndarray,
        df_props: pd.DataFrame,
        cell_type_idx: int,
        cell_type_name: str,
        output_dir: str,
        proportion_threshold: float = 0.05,
        global_embedding: Optional[np.ndarray] = None,
        latent_combination_method: str = 'pca'  # NEW PARAMETER
    ):
        """
        Visualize cell-type-specific states on a CONSISTENT UMAP embedding of ALL spots.
        Spots below the proportion threshold are shown in grey.
        
        This ensures all cell type UMAPs have the same layout for comparison.
        
        Args:
            gamma_states: Full gamma tensor [n_spots, n_cell_types, n_latent]
            df_props: Proportions dataframe
            cell_type_idx: Index of cell type to visualize
            cell_type_name: Name for file saving
            output_dir: Output directory
            proportion_threshold: Threshold for "present" vs "absent"
            global_embedding: Pre-computed UMAP embedding [n_spots, 2] for consistency
            latent_combination_method: How to combine latent dimensions:
                - 'pca': Use first PC of all latents (captures max variance)
                - 'sum': Simple sum of all latent dimensions
                - 'norm': L2 norm (euclidean distance from origin)
                - 'weighted': Variance-weighted combination
                - 'latent0': Use only first latent dimension (original behavior)
            
        Returns:
            global_embedding: The UMAP embedding used (for reuse across cell types)
        """
        from sklearn.decomposition import PCA
        
        # Extract states for this cell type
        ct_gamma = gamma_states[:, cell_type_idx, :]  # [n_spots, n_latent]
        ct_props = df_props.iloc[:, cell_type_idx].values  # [n_spots]
        
        n_spots = gamma_states.shape[0]
        n_latent = ct_gamma.shape[1]
        
        # Create mask for spots where this cell type is present
        mask_present = ct_props >= proportion_threshold
        mask_absent = ~mask_present
        
        n_present = mask_present.sum()
        n_absent = mask_absent.sum()
        
        # =====================================================================
        # NEW: Combine all latent dimensions into a single score
        # =====================================================================
        ct_gamma_filtered = ct_gamma[mask_present]
        
        if latent_combination_method == 'pca' and n_present > 1:
            # Use PCA to find the direction of maximum variance
            pca = PCA(n_components=1)
            pca.fit(ct_gamma_filtered)
            latent_score = pca.transform(ct_gamma).flatten()
            score_name = "PC1 (all latents)"
            print(f"  -> {cell_type_name} (full): PCA explained variance = {pca.explained_variance_ratio_[0]:.2%}")
            
        elif latent_combination_method == 'sum':
            # Simple sum of all latent dimensions
            latent_score = ct_gamma.sum(axis=1)
            score_name = "Sum of Latents"
            
        elif latent_combination_method == 'norm':
            # L2 norm - distance from origin in latent space
            latent_score = np.linalg.norm(ct_gamma, axis=1)
            score_name = "Latent Norm"
            
        elif latent_combination_method == 'weighted' and n_present > 1:
            # Weight by variance of each dimension (more variable = more important)
            variances = ct_gamma_filtered.var(axis=0)
            variances = variances / (variances.sum() + 1e-8)  # Normalize weights
            latent_score = (ct_gamma * variances).sum(axis=1)
            score_name = "Variance-weighted"
            
        else:  # 'latent0' or fallback
            # Original behavior: first dimension only
            latent_score = ct_gamma[:, 0]
            score_name = "Latent_0"
        
        # Print score range for comparison
        if n_present > 0:
            score_present = latent_score[mask_present]
            score_range = score_present.max() - score_present.min()
            print(f"  -> {cell_type_name} (full): {score_name} range = [{score_present.min():.2f}, {score_present.max():.2f}] (span: {score_range:.2f})")
        # =====================================================================
        
        # Compute global embedding if not provided
        # Use ALL latent dimensions flattened for a consistent structure
        if global_embedding is None:
            print("  -> Computing global UMAP embedding for all spots...")
            # Flatten all cell type states into one vector per spot
            all_latent = gamma_states.reshape(n_spots, -1)  # [n_spots, n_cell_types * n_latent]
            
            if HAS_UMAP and n_spots > 30:
                reducer = umap.UMAP(
                    n_neighbors=min(30, n_spots - 1),
                    min_dist=0.3,
                    random_state=42,
                    metric='euclidean'
                )
                global_embedding = reducer.fit_transform(all_latent)
                method_name = "UMAP"
            else:
                reducer = PCA(n_components=2)
                global_embedding = reducer.fit_transform(all_latent)
                method_name = "PCA"
        else:
            method_name = "UMAP" if HAS_UMAP else "PCA"
        
        safe_name = str(cell_type_name).replace("/", "_").replace(" ", "_")
        
        # === PLOT 1: Color by Latent State (Grey for absent) ===
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left panel: Color by combined latent score (NOT just first dimension)
        ax1 = axes[0]
        
        # Plot absent spots first (grey, background)
        if n_absent > 0:
            ax1.scatter(
                global_embedding[mask_absent, 0],
                global_embedding[mask_absent, 1],
                c='lightgrey',
                s=8,
                alpha=0.3,
                label=f'Absent (n={n_absent})',
                edgecolors='none'
            )
        
        # Plot present spots with color
        if n_present > 0:
            # USE THE COMBINED LATENT SCORE
            latent_values = latent_score[mask_present]
            
            sc1 = ax1.scatter(
                global_embedding[mask_present, 0],
                global_embedding[mask_present, 1],
                c=latent_values,
                cmap='coolwarm',
                s=15,
                alpha=0.8,
                edgecolors='none'
            )
            plt.colorbar(sc1, ax=ax1, label=score_name, shrink=0.8)
        
        ax1.set_title(f"{cell_type_name}: State Variation ({score_name})\n(Grey = proportion < {proportion_threshold:.0%})")
        ax1.set_xlabel(f"{method_name} 1")
        ax1.set_ylabel(f"{method_name} 2")
        ax1.legend(loc='upper right', fontsize=8)
        
        # Right panel: Color by proportion intensity
        ax2 = axes[1]
        
        # Plot absent spots (grey)
        if n_absent > 0:
            ax2.scatter(
                global_embedding[mask_absent, 0],
                global_embedding[mask_absent, 1],
                c='lightgrey',
                s=8,
                alpha=0.3,
                label=f'Absent (n={n_absent})',
                edgecolors='none'
            )
        
        # Plot present spots colored by proportion
        if n_present > 0:
            prop_values = ct_props[mask_present]
            
            sc2 = ax2.scatter(
                global_embedding[mask_present, 0],
                global_embedding[mask_present, 1],
                c=prop_values,
                cmap='viridis',
                s=15,
                alpha=0.8,
                vmin=proportion_threshold,
                vmax=min(1.0, ct_props.max()),
                edgecolors='none'
            )
            plt.colorbar(sc2, ax=ax2, label='Proportion', shrink=0.8)
        
        ax2.set_title(f"{cell_type_name}: Proportion Intensity\n(n_present={n_present}, n_absent={n_absent})")
        ax2.set_xlabel(f"{method_name} 1")
        ax2.set_ylabel(f"{method_name} 2")
        ax2.legend(loc='upper right', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"continuum_full_{safe_name}.png"), dpi=300)
        plt.close()
        
        print(f"  -> Saved continuum_full_{safe_name}.png (present: {n_present}, absent: {n_absent})")
        
        return global_embedding
    
    @staticmethod
    def plot_cell_type_marker_genes(
        df_props: pd.DataFrame,
        X_st: np.ndarray,
        gene_names: List[str],
        reference_profiles: np.ndarray,
        coords: np.ndarray,
        output_dir: str,
        n_top_genes: int = 5,
        coords_full: Optional[np.ndarray] = None,
        matched_mask: Optional[np.ndarray] = None,
        presence_threshold: float = 0.05
    ):
        """
        Visualize top marker genes for each cell type across spatial locations.
        
        For each cell type:
        1. Identifies top N marker genes (highest in reference profile)
        2. Creates spatial maps showing expression of each marker gene
        3. Shows which spots have high expression of cell-type-specific markers
        
        Args:
            df_props: Proportions dataframe [n_spots, n_cell_types]
            X_st: Spatial expression matrix [n_spots, n_genes]
            gene_names: List of gene names
            reference_profiles: Reference profiles [n_cell_types, n_genes]
            coords: Spatial coordinates [n_spots, 2]
            output_dir: Output directory
            n_top_genes: Number of top marker genes to show (default: 5)
            coords_full: All coordinates for background
            matched_mask: Boolean mask for matched spots
            presence_threshold: Threshold for cell type presence
        """
        from matplotlib.patches import RegularPolygon
        from matplotlib.collections import PatchCollection
        import matplotlib.gridspec as gridspec
        
        cell_types = df_props.columns.tolist()
        n_cell_types = len(cell_types)
        n_spots = X_st.shape[0]
        
        # Calculate hexagon radius
        coords_for_radius = coords_full if coords_full is not None else coords
        hex_radius = VisualizationUtils._calculate_hex_radius(coords_for_radius, scale_factor=0.55)
        
        # Calculate axis limits
        coords_for_limits = coords_full if coords_full is not None else coords
        x_min = coords_for_limits[:, 0].min() - hex_radius * 2
        x_max = coords_for_limits[:, 0].max() + hex_radius * 2
        y_min = coords_for_limits[:, 1].min() - hex_radius * 2
        y_max = coords_for_limits[:, 1].max() + hex_radius * 2
        
        # Normalize spatial expression to CPM for visualization
        X_st_cpm = X_st / (X_st.sum(axis=1, keepdims=True) + 1e-8) * 1e6
        X_st_log = np.log1p(X_st_cpm)
        
        # Helper function to add background spots
        def add_background_spots(ax):
            if coords_full is None or matched_mask is None:
                return
            unmatched_mask = ~matched_mask
            if not unmatched_mask.any():
                return
            unmatched_coords = coords_full[unmatched_mask]
            patches = []
            for x, y in unmatched_coords:
                hexagon = RegularPolygon(
                    (x, y), numVertices=6, radius=hex_radius,
                    orientation=HEXAGON_ORIENTATION
                )
                patches.append(hexagon)
            collection = PatchCollection(
                patches, facecolors='lightgrey', edgecolors='none', alpha=0.3
            )
            ax.add_collection(collection)
        
        # Process each cell type
        for ct_idx, ct_name in enumerate(cell_types):
            safe_name = str(ct_name).replace("/", "_").replace(" ", "_")
            
            # Get reference profile for this cell type
            ct_profile = reference_profiles[ct_idx]
            
            # Find top marker genes (highest expression in reference)
            # Also consider specificity: high in this type, low in others
            other_profiles = np.delete(reference_profiles, ct_idx, axis=0)
            other_max = other_profiles.max(axis=0)
            
            # Specificity score: ratio of this type's expression to max of others
            specificity = ct_profile / (other_max + 1e-8)
            
            # Combined score: expression * specificity
            marker_score = ct_profile * np.log1p(specificity)
            
            # Get top genes
            top_gene_indices = np.argsort(marker_score)[::-1][:n_top_genes]
            top_gene_names = [gene_names[i] for i in top_gene_indices]
            
            print(f"\n{ct_name} - Top {n_top_genes} marker genes:")
            for i, (gene_idx, gene_name) in enumerate(zip(top_gene_indices, top_gene_names)):
                print(f"  {i+1}. {gene_name} (score: {marker_score[gene_idx]:.4f}, "
                      f"expr: {ct_profile[gene_idx]:.4f}, spec: {specificity[gene_idx]:.2f})")
            
            # Create figure with subplots for each marker gene + proportion map
            n_cols = 3
            n_rows = 2
            fig = plt.figure(figsize=(5 * n_cols, 4.5 * n_rows))
            gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, wspace=0.3, hspace=0.4)
            
            # First subplot: Cell type proportion map
            ax0 = fig.add_subplot(gs[0, 0])
            ct_props = df_props[ct_name].values
            above_threshold = ct_props >= presence_threshold
            
            add_background_spots(ax0)
            
            # Plot spots below threshold in grey
            patches_below = []
            for j, (x, y) in enumerate(coords):
                if not above_threshold[j]:
                    hexagon = RegularPolygon(
                        (x, y), numVertices=6, radius=hex_radius,
                        orientation=HEXAGON_ORIENTATION
                    )
                    patches_below.append(hexagon)
            
            if patches_below:
                collection_below = PatchCollection(
                    patches_below, facecolors='lightgrey', edgecolors='none', alpha=0.4
                )
                ax0.add_collection(collection_below)
            
            # Plot spots above threshold with color
            patches_above = []
            colors_above = []
            values_above = ct_props[above_threshold]
            if len(values_above) > 0:
                vmin, vmax = values_above.min(), values_above.max()
                if vmax - vmin < 1e-8:
                    vmax = vmin + 1e-8
                norm = plt.Normalize(vmin=vmin, vmax=vmax)
                cmap = plt.cm.Reds
                
                for j, (x, y) in enumerate(coords):
                    if above_threshold[j]:
                        hexagon = RegularPolygon(
                            (x, y), numVertices=6, radius=hex_radius,
                            orientation=HEXAGON_ORIENTATION
                        )
                        patches_above.append(hexagon)
                        colors_above.append(cmap(norm(ct_props[j])))
                
                if patches_above:
                    collection_above = PatchCollection(
                        patches_above, facecolors=colors_above, edgecolors='none'
                    )
                    ax0.add_collection(collection_above)
                
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                plt.colorbar(sm, ax=ax0, shrink=0.7, label='Proportion')
            
            ax0.set_xlim(x_min, x_max)
            ax0.set_ylim(y_min, y_max)
            ax0.set_aspect('equal')
            ax0.invert_yaxis()
            ax0.axis('off')
            ax0.set_title(f"{ct_name}\nProportion", fontsize=12, fontweight='bold')
            
            # Remaining subplots: Top marker genes
            for gene_plot_idx, (gene_idx, gene_name) in enumerate(zip(top_gene_indices, top_gene_names)):
                row = (gene_plot_idx + 1) // n_cols
                col = (gene_plot_idx + 1) % n_cols
                ax = fig.add_subplot(gs[row, col])
                
                # Get gene expression
                gene_expr = X_st_log[:, gene_idx]
                
                add_background_spots(ax)
                
                # Create patches for all spots
                patches = []
                colors = []
                
                # Normalize expression for colormap
                expr_min, expr_max = gene_expr.min(), gene_expr.max()
                if expr_max - expr_min < 1e-8:
                    expr_max = expr_min + 1e-8
                norm = plt.Normalize(vmin=expr_min, vmax=expr_max)
                cmap = plt.cm.viridis
                
                for j, (x, y) in enumerate(coords):
                    hexagon = RegularPolygon(
                        (x, y), numVertices=6, radius=hex_radius,
                        orientation=HEXAGON_ORIENTATION
                    )
                    patches.append(hexagon)
                    
                    # Color by expression, but highlight spots where cell type is present
                    if above_threshold[j]:
                        # Full opacity for spots with this cell type
                        rgba = list(cmap(norm(gene_expr[j])))
                        rgba[3] = 1.0
                    else:
                        # Lower opacity for spots without this cell type
                        rgba = list(cmap(norm(gene_expr[j])))
                        rgba[3] = 0.3
                    colors.append(rgba)
                
                collection = PatchCollection(patches, facecolors=colors, edgecolors='none')
                ax.add_collection(collection)
                
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_aspect('equal')
                ax.invert_yaxis()
                ax.axis('off')
                
                # Add colorbar
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                plt.colorbar(sm, ax=ax, shrink=0.7, label='log(CPM+1)')
                
                ax.set_title(f"{gene_name}\n(Marker #{gene_plot_idx+1})", fontsize=11, fontweight='bold')
            
            plt.suptitle(f"{ct_name} - Top {n_top_genes} Marker Genes", 
                        fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f"marker_genes_{safe_name}.png"),
                dpi=300, bbox_inches='tight'
            )
            plt.close()
            print(f"  -> Saved marker_genes_{safe_name}.png")
        
        # Create summary figure: All cell types with their top marker
        print("\nGenerating summary marker gene map...")
        
        n_cols_summary = min(4, n_cell_types)
        n_rows_summary = math.ceil(n_cell_types / n_cols_summary)
        fig, axes = plt.subplots(
            n_rows_summary, n_cols_summary,
            figsize=(5 * n_cols_summary, 4.5 * n_rows_summary)
        )
        if n_cell_types == 1:
            axes = np.array([[axes]])
        axes = axes.flatten()
        
        for ct_idx, ct_name in enumerate(cell_types):
            ax = axes[ct_idx]
            
            # Get top marker gene for this cell type
            ct_profile = reference_profiles[ct_idx]
            other_profiles = np.delete(reference_profiles, ct_idx, axis=0)
            other_max = other_profiles.max(axis=0)
            specificity = ct_profile / (other_max + 1e-8)
            marker_score = ct_profile * np.log1p(specificity)
            top_gene_idx = np.argmax(marker_score)
            top_gene_name = gene_names[top_gene_idx]
            
            # Get expression and proportion
            gene_expr = X_st_log[:, top_gene_idx]
            ct_props = df_props[ct_name].values
            above_threshold = ct_props >= presence_threshold
            
            add_background_spots(ax)
            
            # Create patches
            patches = []
            colors = []
            
            expr_min, expr_max = gene_expr.min(), gene_expr.max()
            if expr_max - expr_min < 1e-8:
                expr_max = expr_min + 1e-8
            norm = plt.Normalize(vmin=expr_min, vmax=expr_max)
            cmap = plt.cm.plasma
            
            for j, (x, y) in enumerate(coords):
                hexagon = RegularPolygon(
                    (x, y), numVertices=6, radius=hex_radius,
                    orientation=HEXAGON_ORIENTATION
                )
                patches.append(hexagon)
                
                if above_threshold[j]:
                    rgba = list(cmap(norm(gene_expr[j])))
                    rgba[3] = 1.0
                else:
                    rgba = [0.8, 0.8, 0.8, 0.3]  # Grey for absent
                colors.append(rgba)
            
            collection = PatchCollection(patches, facecolors=colors, edgecolors='none')
            ax.add_collection(collection)
            
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.axis('off')
            
            n_present = above_threshold.sum()
            ax.set_title(f"{ct_name}\n{top_gene_name} (n={n_present})", 
                        fontsize=11, fontweight='bold')
            
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, shrink=0.7)
        
        # Hide empty subplots
        for j in range(ct_idx + 1, len(axes)):
            axes[j].axis('off')
        
        plt.suptitle("Top Marker Gene Expression per Cell Type", 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "marker_genes_summary.png"),
            dpi=300, bbox_inches='tight'
        )
        plt.close()
        print("  -> Saved marker_genes_summary.png")
    @staticmethod
    def plot_marker_gene_dotplot(
        X_sc: np.ndarray,
        cell_types: np.ndarray,
        gene_names: List[str],
        cell_type_names: List[str],
        reference_profiles: np.ndarray,
        output_path: str,
        n_top_genes: int = 5,
        expression_threshold: float = 0.0,
        # === NEW CUSTOMIZATION PARAMETERS ===
        figsize: Tuple[float, float] = None,  # Auto-calculated if None
        dot_size_scale: float = 600,          # Maximum dot size
        dot_min_size: float = 5,              # Minimum visible dot size
        cmap: str = 'Reds',                   # Colormap for expression
        edgecolor: str = 'black',             # Dot edge color
        linewidth: float = 0.3,               # Dot edge width
        # Font sizes
        title_fontsize: int = 24,
        axis_label_fontsize: int = 24,
        xtick_fontsize: int = 24,
        ytick_fontsize: int = 24,
        group_label_fontsize: int = 24,
        legend_fontsize: int = 24,
        colorbar_label_fontsize: int = 24,
        # Rotation
        xtick_rotation: int = 90,
        group_label_rotation: int = 90,
        # Spacing
        wspace: float = 0.3,                  # Width space
        hspace: float = 0.3,                  # Height space
        # Grid
        show_grid: bool = True,
        grid_alpha: float = 0.2,
        # Separators
        show_separators: bool = True,
        separator_color: str = 'grey',
        separator_linewidth: float = 0.5,
    ):
        """
        Create a standard scRNA-seq marker gene dot plot.
        
        This is the standard Seurat/Scanpy style dot plot using SINGLE-CELL data:
        - Dot size = fraction of CELLS in that type expressing the gene
        - Dot color = mean expression in CELLS of that type
        
        The plot should show a BLOCK-DIAGONAL pattern where each cell type's
        marker genes (columns) have large dots in the row for that cell type.
        
        Args:
            X_sc: Single-cell expression matrix [n_cells, n_genes]
            cell_types: Cell type indices for each cell [n_cells]
            gene_names: List of gene names
            cell_type_names: List of cell type names
            reference_profiles: Reference profiles [n_cell_types, n_genes]
            output_path: Path to save figure
            n_top_genes: Number of top genes per cell type
            expression_threshold: Threshold for "expressing" (raw count > threshold)
        """
        n_cells, n_genes_total = X_sc.shape
        n_cell_types = len(cell_type_names)
        
        # Normalize to log CPM for mean expression calculation
        lib_sizes = X_sc.sum(axis=1, keepdims=True)
        X_cpm = X_sc / (lib_sizes + 1e-8) * 1e6
        X_log = np.log1p(X_cpm)
        
        # Collect top markers for each cell type IN ORDER
        # This ensures genes are grouped by their source cell type
        all_markers = []  # List of (gene_name, source_cell_type_idx)
        gene_to_source = {}  # Track which cell type each gene is a marker for
        
        for ct_idx, ct_name in enumerate(cell_type_names):
            ct_profile = reference_profiles[ct_idx]
            other_profiles = np.delete(reference_profiles, ct_idx, axis=0)
            other_max = other_profiles.max(axis=0) if len(other_profiles) > 0 else np.zeros_like(ct_profile)
            
            # Specificity score
            specificity = ct_profile / (other_max + 1e-8)
            marker_score = ct_profile * np.log1p(specificity)
            
            # Get top genes for this cell type
            top_indices = np.argsort(marker_score)[::-1]
            
            # Add genes that aren't already in the list
            added = 0
            for idx in top_indices:
                gene = gene_names[idx]
                if gene not in gene_to_source and added < n_top_genes:
                    all_markers.append((gene, ct_idx))
                    gene_to_source[gene] = ct_idx
                    added += 1
                if added >= n_top_genes:
                    break
        
        n_markers = len(all_markers)
        marker_genes = [m[0] for m in all_markers]
        marker_sources = [m[1] for m in all_markers]
        
        # Compute dot plot data FROM SINGLE CELLS
        dot_sizes = np.zeros((n_cell_types, n_markers))
        dot_colors = np.zeros((n_cell_types, n_markers))
        
        for ct_idx in range(n_cell_types):
            # Get cells of this type
            ct_mask = (cell_types == ct_idx)
            n_cells_ct = ct_mask.sum()
            
            if n_cells_ct == 0:
                continue
            
            X_ct_log = X_log[ct_mask]  # Cells of this type (log CPM)
            X_ct_raw = X_sc[ct_mask]   # Raw counts for percent expressing
            
            for gene_idx, gene in enumerate(marker_genes):
                try:
                    gene_col_idx = gene_names.index(gene)
                except ValueError:
                    continue
                
                # Mean expression in cells of this type (log scale)
                mean_expr = X_ct_log[:, gene_col_idx].mean()
                
                # Fraction of cells expressing (raw count > threshold)
                frac_expressing = (X_ct_raw[:, gene_col_idx] > expression_threshold).mean()
                
                dot_sizes[ct_idx, gene_idx] = frac_expressing
                dot_colors[ct_idx, gene_idx] = mean_expr
        
        # Create figure
        if figsize is None:
            fig_width = max(12, n_markers * 0.5 + 4)
            fig_height = max(6, n_cell_types * 0.5 + 2)
            figsize = (fig_width, fig_height)
        fig, ax = plt.subplots(figsize=figsize)
        
        # Color normalization
        vmin = 0
        vmax = dot_colors.max() if dot_colors.max() > 0 else 1
        
        # Create dot plot
        for ct_idx in range(n_cell_types):
            for gene_idx in range(n_markers):
                size = dot_sizes[ct_idx, gene_idx]
                color = dot_colors[ct_idx, gene_idx]
                
                # Scale size for visibility
                plot_size = size * dot_size_scale
                
                if plot_size > dot_min_size:  # Only plot if visible
                    ax.scatter(
                        gene_idx, ct_idx,
                        s=plot_size,
                        c=[color],
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        edgecolors=edgecolor,
                        linewidths=linewidth,
                        zorder=3
                    )
        
        # Add cell type grouping labels on top
        # Draw brackets/separators between gene groups
        current_source = -1
        group_starts = []
        for gene_idx, source_ct in enumerate(marker_sources):
            if source_ct != current_source:
                group_starts.append((gene_idx, source_ct))
                current_source = source_ct
        
        # Add vertical separators between groups
        if show_separators:
            for i, (start_idx, _) in enumerate(group_starts[1:], 1):
                ax.axvline(x=start_idx - 0.5, color=separator_color, linestyle='-', 
                          linewidth=separator_linewidth, alpha=0.5)
        
        # Add group labels on top
        for i, (start_idx, source_ct) in enumerate(group_starts):
            if i < len(group_starts) - 1:
                end_idx = group_starts[i + 1][0] - 1
            else:
                end_idx = n_markers - 1
            
            mid_x = (start_idx + end_idx) / 2
            ax.text(
                mid_x, -0.8,  # Position above the plot (negative y after inversion)
                cell_type_names[source_ct],
                ha='center', va='bottom',
                fontsize=group_label_fontsize,
                fontweight='bold',
                rotation=group_label_rotation
            )
        
        # Invert y-axis so first cell type is at top (diagonal from top-left to bottom-right)
        ax.invert_yaxis()
        
        # Set axis limits to remove extra space
        ax.set_xlim(-0.5, n_markers - 0.5)
        ax.set_ylim(n_cell_types - 0.5, -0.5)  # Inverted: larger value first
        
        # Formatting
        ax.set_xticks(range(n_markers))
        ax.set_xticklabels(
            marker_genes,
            rotation=xtick_rotation,
            ha='center',
            fontsize=xtick_fontsize
        )
        ax.set_yticks(range(n_cell_types))
        ax.set_yticklabels(
            cell_type_names,
            fontsize=ytick_fontsize
        )
        
        # Add grid
        ax.set_axisbelow(True)
        if show_grid:
            ax.grid(True, alpha=grid_alpha, linestyle=':')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('Mean Expression\n(log CPM)', fontsize=colorbar_label_fontsize)
        
        # Add size legend
        legend_fracs = [0.25, 0.50, 0.75, 1.00]
        legend_handles = []
        for frac in legend_fracs:
            handle = ax.scatter(
                [], [], 
                s=frac * dot_size_scale, 
                c='grey', 
                edgecolors=edgecolor, 
                linewidths=linewidth,
                label=f'{int(frac*100)}%'
            )
            legend_handles.append(handle)
        
        ax.legend(
            handles=legend_handles,
            title='% Expressing',
            loc='upper left',
            bbox_to_anchor=(1.12, 1),
            frameon=True,
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize + 1
        )
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  -> Saved {os.path.basename(output_path)}")
        print(f"     Expected diagonal pattern: Large red dots where row cell type matches column gene group")
    
# ==============================================================================
# SECTION 12.7: MARKER GENE AND DIFFERENTIAL EXPRESSION ANALYSIS
# ==============================================================================

class MarkerGeneAnalysis:
    """
    Comprehensive marker gene identification and differential expression analysis.
    
    Generates two types of output:
    1. marker_gene_rankings/ - Per cell type ranking of all genes with multiple scores
    2. differential_expression/ - Per cell type DE analysis (this type vs all others)
    """
    
    @staticmethod
    def compute_marker_scores(
        reference_profiles: np.ndarray,
        gene_names: List[str],
        cell_type_names: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute comprehensive marker gene scores for each cell type.
        
        Scoring methods included:
        1. Expression level: Mean expression in reference profile
        2. Specificity: Ratio of expression in this type vs max of others
        3. Fold change: Log2 fold change vs mean of other types
        4. Combined score: Expression × log(1 + specificity)
        5. Gini coefficient: How specific is expression to this type
        
        Args:
            reference_profiles: [n_cell_types, n_genes] normalized profiles
            gene_names: List of gene names
            cell_type_names: List of cell type names
            
        Returns:
            Dictionary mapping cell type names to DataFrames with gene rankings
        """
        n_cell_types, n_genes = reference_profiles.shape
        results = {}
        
        for ct_idx, ct_name in enumerate(cell_type_names):
            # Get this cell type's profile
            ct_profile = reference_profiles[ct_idx]
            
            # Get other cell types' profiles
            other_profiles = np.delete(reference_profiles, ct_idx, axis=0)
            other_max = other_profiles.max(axis=0)
            other_mean = other_profiles.mean(axis=0)
            other_min = other_profiles.min(axis=0)
            
            # 1. Raw expression level
            expression = ct_profile.copy()
            
            # 2. Specificity score (ratio to max of others)
            specificity = ct_profile / (other_max + 1e-10)
            
            # 3. Log2 fold change vs mean of others
            log2_fc = np.log2((ct_profile + 1e-10) / (other_mean + 1e-10))
            
            # 4. Combined score (expression × log(1 + specificity))
            combined_score = ct_profile * np.log1p(specificity)
            
            # 5. Gini-like coefficient for this gene across cell types
            gini_scores = np.zeros(n_genes)
            for g in range(n_genes):
                gene_expr = reference_profiles[:, g]
                if gene_expr.sum() > 0:
                    # Normalized Gini: higher = more specific
                    sorted_expr = np.sort(gene_expr)
                    n = len(sorted_expr)
                    indices = np.arange(1, n + 1)
                    gini = (2 * np.sum(indices * sorted_expr) / (n * np.sum(sorted_expr))) - (n + 1) / n
                    # Weight by whether this type has high expression
                    rank_in_type = np.where(np.argsort(gene_expr)[::-1] == ct_idx)[0][0]
                    if rank_in_type == 0:  # Highest in this type
                        gini_scores[g] = gini
                    else:
                        gini_scores[g] = gini * (1 - rank_in_type / n)
            
            # 6. Percent of total expression from this cell type
            total_expr = reference_profiles.sum(axis=0)
            pct_expression = ct_profile / (total_expr + 1e-10) * 100
            
            # 7. Rank among cell types (1 = highest)
            ranks = np.zeros(n_genes, dtype=int)
            for g in range(n_genes):
                gene_expr = reference_profiles[:, g]
                sorted_indices = np.argsort(gene_expr)[::-1]
                ranks[g] = np.where(sorted_indices == ct_idx)[0][0] + 1
            
            # 8. Z-score of expression
            mean_expr = reference_profiles.mean(axis=0)
            std_expr = reference_profiles.std(axis=0)
            z_score = (ct_profile - mean_expr) / (std_expr + 1e-10)
            
            # Create DataFrame
            df = pd.DataFrame({
                'gene': gene_names,
                'expression': expression,
                'specificity': specificity,
                'log2_fc_vs_others': log2_fc,
                'combined_score': combined_score,
                'gini_specificity': gini_scores,
                'pct_total_expression': pct_expression,
                'rank_among_types': ranks,
                'z_score': z_score,
                'max_other_expression': other_max,
                'mean_other_expression': other_mean,
                'is_top_expressed': (ranks == 1).astype(int)
            })
            
            # Sort by combined score
            df = df.sort_values('combined_score', ascending=False)
            df['marker_rank'] = range(1, len(df) + 1)
            
            # Reorder columns
            df = df[['marker_rank', 'gene', 'combined_score', 'expression', 
                     'specificity', 'log2_fc_vs_others', 'z_score',
                     'gini_specificity', 'pct_total_expression', 
                     'rank_among_types', 'is_top_expressed',
                     'max_other_expression', 'mean_other_expression']]
            
            results[ct_name] = df
            
        return results
    
    @staticmethod
    def compute_differential_expression(
        X_sc: np.ndarray,
        cell_types: np.ndarray,
        gene_names: List[str],
        cell_type_names: List[str],
        min_cells: int = 10,
        min_pct_expressing: float = 0.1
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute differential expression: each cell type vs all others combined.
        
        Uses Wilcoxon rank-sum test (Mann-Whitney U) for statistical testing.
        
        Args:
            X_sc: [n_cells, n_genes] raw count matrix
            cell_types: [n_cells] cell type indices
            gene_names: List of gene names
            cell_type_names: List of cell type names
            min_cells: Minimum cells required per group
            min_pct_expressing: Minimum percent of cells expressing gene
            
        Returns:
            Dictionary mapping cell type names to DE result DataFrames
        """
        from scipy.stats import mannwhitneyu, ranksums
        
        n_cells, n_genes = X_sc.shape
        n_cell_types = len(cell_type_names)
        
        # Normalize to CPM and log transform for DE
        lib_sizes = X_sc.sum(axis=1, keepdims=True)
        X_cpm = X_sc / (lib_sizes + 1e-8) * 1e6
        X_log = np.log1p(X_cpm)
        
        results = {}
        
        for ct_idx, ct_name in enumerate(cell_type_names):
            print(f"  Computing DE for {ct_name}...")
            
            # Split cells
            mask_this = (cell_types == ct_idx)
            mask_other = ~mask_this
            
            n_this = mask_this.sum()
            n_other = mask_other.sum()
            
            if n_this < min_cells or n_other < min_cells:
                print(f"    -> Skipping: insufficient cells (this={n_this}, other={n_other})")
                continue
            
            X_this = X_log[mask_this]
            X_other = X_log[mask_other]
            
            # Also get raw counts for percent expressing
            X_raw_this = X_sc[mask_this]
            X_raw_other = X_sc[mask_other]
            
            # Initialize result arrays
            log2_fc = np.zeros(n_genes)
            p_values = np.ones(n_genes)
            mean_this = np.zeros(n_genes)
            mean_other = np.zeros(n_genes)
            pct_this = np.zeros(n_genes)
            pct_other = np.zeros(n_genes)
            
            for g in range(n_genes):
                expr_this = X_this[:, g]
                expr_other = X_other[:, g]
                
                # Mean expression (log scale)
                mean_this[g] = expr_this.mean()
                mean_other[g] = expr_other.mean()
                
                # Log2 fold change
                log2_fc[g] = mean_this[g] - mean_other[g]  # Already in log scale
                
                # Percent expressing (from raw counts)
                pct_this[g] = (X_raw_this[:, g] > 0).mean() * 100
                pct_other[g] = (X_raw_other[:, g] > 0).mean() * 100
                
                # Statistical test (only if enough cells express)
                if pct_this[g] >= min_pct_expressing * 100 or pct_other[g] >= min_pct_expressing * 100:
                    try:
                        # Mann-Whitney U test (two-sided)
                        _, p_values[g] = mannwhitneyu(
                            expr_this, expr_other, 
                            alternative='two-sided',
                            use_continuity=True
                        )
                    except Exception:
                        p_values[g] = 1.0
            
            # Multiple testing correction (Benjamini-Hochberg)
            p_adjusted = MarkerGeneAnalysis._benjamini_hochberg(p_values)
            
            # Compute additional metrics
            # Effect size (Cohen's d approximation)
            pooled_std = np.sqrt(
                (X_this.var(axis=0) * (n_this - 1) + X_other.var(axis=0) * (n_other - 1)) 
                / (n_this + n_other - 2) + 1e-10
            )
            cohens_d = (mean_this - mean_other) / pooled_std
            
            # Create DataFrame
            df = pd.DataFrame({
                'gene': gene_names,
                'log2_fold_change': log2_fc,
                'mean_expr_this': mean_this,
                'mean_expr_other': mean_other,
                'pct_expressing_this': pct_this,
                'pct_expressing_other': pct_other,
                'pct_diff': pct_this - pct_other,
                'p_value': p_values,
                'p_adjusted': p_adjusted,
                'neg_log10_pval': -np.log10(p_values + 1e-300),
                'neg_log10_padj': -np.log10(p_adjusted + 1e-300),
                'cohens_d': cohens_d,
                'is_significant': (p_adjusted < 0.05).astype(int),
                'is_upregulated': ((p_adjusted < 0.05) & (log2_fc > 0)).astype(int),
                'is_downregulated': ((p_adjusted < 0.05) & (log2_fc < 0)).astype(int)
            })
            
            # Add regulation direction
            df['regulation'] = 'NS'  # Not significant
            df.loc[(df['p_adjusted'] < 0.05) & (df['log2_fold_change'] > 0.25), 'regulation'] = 'UP'
            df.loc[(df['p_adjusted'] < 0.05) & (df['log2_fold_change'] < -0.25), 'regulation'] = 'DOWN'
            
            # Sort by significance and fold change
            df['sort_key'] = -df['neg_log10_padj'] * np.sign(df['log2_fold_change']) * np.abs(df['log2_fold_change'])
            df = df.sort_values('sort_key')
            df = df.drop('sort_key', axis=1)
            
            # Add rank
            df['de_rank'] = range(1, len(df) + 1)
            
            # Reorder columns
            df = df[['de_rank', 'gene', 'log2_fold_change', 'p_value', 'p_adjusted',
                     'neg_log10_pval', 'neg_log10_padj', 'regulation',
                     'mean_expr_this', 'mean_expr_other', 
                     'pct_expressing_this', 'pct_expressing_other', 'pct_diff',
                     'cohens_d', 'is_significant', 'is_upregulated', 'is_downregulated']]
            
            # Add metadata
            df.attrs['n_cells_this'] = n_this
            df.attrs['n_cells_other'] = n_other
            df.attrs['n_significant'] = (df['is_significant'] == 1).sum()
            df.attrs['n_upregulated'] = (df['is_upregulated'] == 1).sum()
            df.attrs['n_downregulated'] = (df['is_downregulated'] == 1).sum()
            
            results[ct_name] = df
            
            print(f"    -> Found {df.attrs['n_significant']} significant genes "
                  f"({df.attrs['n_upregulated']} up, {df.attrs['n_downregulated']} down)")
            
        return results
    
    @staticmethod
    def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
        """Apply Benjamini-Hochberg FDR correction."""
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_pvals = p_values[sorted_indices]
        
        # BH adjustment
        adjusted = np.zeros(n)
        for i, idx in enumerate(sorted_indices):
            adjusted[idx] = sorted_pvals[i] * n / (i + 1)
        
        # Ensure monotonicity (cumulative minimum from the end)
        adjusted_sorted = adjusted[sorted_indices]
        for i in range(n - 2, -1, -1):
            adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])
        adjusted[sorted_indices] = adjusted_sorted
        
        # Clip to [0, 1]
        adjusted = np.clip(adjusted, 0, 1)
        
        return adjusted
    
    @staticmethod
    def save_results(
        marker_results: Dict[str, pd.DataFrame],
        de_results: Dict[str, pd.DataFrame],
        output_dir: str
    ):
        """
        Save marker gene rankings and DE results to CSV files.
        
        Creates:
        - output_dir/marker_gene_rankings/{cell_type}_markers.csv
        - output_dir/differential_expression/{cell_type}_de.csv
        """
        # Create subdirectories
        marker_dir = os.path.join(output_dir, "marker_gene_rankings")
        de_dir = os.path.join(output_dir, "differential_expression")
        os.makedirs(marker_dir, exist_ok=True)
        os.makedirs(de_dir, exist_ok=True)
        
        # Save marker rankings
        print("\nSaving marker gene rankings...")
        for ct_name, df in marker_results.items():
            safe_name = str(ct_name).replace("/", "_").replace(" ", "_").replace(".", "_")
            filepath = os.path.join(marker_dir, f"{safe_name}_markers.csv")
            df.to_csv(filepath, index=False)
            print(f"  - Saved {safe_name}_markers.csv ({len(df)} genes)")
        
        # Save DE results
        print("\nSaving differential expression results...")
        for ct_name, df in de_results.items():
            safe_name = str(ct_name).replace("/", "_").replace(" ", "_").replace(".", "_")
            filepath = os.path.join(de_dir, f"{safe_name}_de.csv")
            
            # Add header comment with metadata
            n_sig = df.attrs.get('n_significant', 'N/A')
            n_up = df.attrs.get('n_upregulated', 'N/A')
            n_down = df.attrs.get('n_downregulated', 'N/A')
            n_this = df.attrs.get('n_cells_this', 'N/A')
            n_other = df.attrs.get('n_cells_other', 'N/A')
            
            df.to_csv(filepath, index=False)
            print(f"  - Saved {safe_name}_de.csv "
                  f"(sig={n_sig}, up={n_up}, down={n_down})")
        
        # Save summary file
        summary_path = os.path.join(output_dir, "gene_analysis_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("GENE ANALYSIS SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("MARKER GENE RANKINGS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Output directory: {marker_dir}\n\n")
            
            for ct_name, df in marker_results.items():
                f.write(f"\n{ct_name}:\n")
                f.write(f"  Total genes ranked: {len(df)}\n")
                top_5 = df.head(5)['gene'].tolist()
                f.write(f"  Top 5 markers: {', '.join(top_5)}\n")
            
            f.write("\n\nDIFFERENTIAL EXPRESSION ANALYSIS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Output directory: {de_dir}\n")
            f.write("Method: Wilcoxon rank-sum test (Mann-Whitney U)\n")
            f.write("Multiple testing correction: Benjamini-Hochberg FDR\n\n")
            
            for ct_name, df in de_results.items():
                n_sig = df.attrs.get('n_significant', 0)
                n_up = df.attrs.get('n_upregulated', 0)
                n_down = df.attrs.get('n_downregulated', 0)
                n_this = df.attrs.get('n_cells_this', 0)
                n_other = df.attrs.get('n_cells_other', 0)
                
                f.write(f"\n{ct_name} vs Rest:\n")
                f.write(f"  Cells in group: {n_this}\n")
                f.write(f"  Cells in rest: {n_other}\n")
                f.write(f"  Significant genes (padj < 0.05): {n_sig}\n")
                f.write(f"    - Upregulated: {n_up}\n")
                f.write(f"    - Downregulated: {n_down}\n")
                
                # Top upregulated
                up_genes = df[df['is_upregulated'] == 1].head(5)['gene'].tolist()
                if up_genes:
                    f.write(f"  Top 5 upregulated: {', '.join(up_genes)}\n")
                
                # Top downregulated  
                down_genes = df[df['is_downregulated'] == 1].tail(5)['gene'].tolist()
                if down_genes:
                    f.write(f"  Top 5 downregulated: {', '.join(down_genes)}\n")
        
        print(f"\nSaved summary to: {summary_path}")
    
    @staticmethod
    def plot_volcano_plots(
        de_results: Dict[str, pd.DataFrame],
        output_dir: str,
        log2fc_threshold: float = 0.5,
        padj_threshold: float = 0.05,
        n_label_genes: int = 10
    ):
        """
        Generate volcano plots for each cell type's DE results.
        
        Args:
            de_results: Dictionary of DE DataFrames
            output_dir: Output directory
            log2fc_threshold: Threshold for highlighting
            padj_threshold: Significance threshold
            n_label_genes: Number of top genes to label
        """
        de_dir = os.path.join(output_dir, "differential_expression")
        os.makedirs(de_dir, exist_ok=True)
        
        print("\nGenerating volcano plots...")
        
        for ct_name, df in de_results.items():
            safe_name = str(ct_name).replace("/", "_").replace(" ", "_").replace(".", "_")
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Define colors based on significance and fold change
            colors = []
            for _, row in df.iterrows():
                if row['p_adjusted'] >= padj_threshold:
                    colors.append('lightgrey')
                elif row['log2_fold_change'] > log2fc_threshold:
                    colors.append('red')
                elif row['log2_fold_change'] < -log2fc_threshold:
                    colors.append('blue')
                else:
                    colors.append('grey')
            
            # Scatter plot
            ax.scatter(
                df['log2_fold_change'],
                df['neg_log10_padj'],
                c=colors,
                s=8,
                alpha=0.6,
                edgecolors='none'
            )
            
            # Add threshold lines
            ax.axhline(y=-np.log10(padj_threshold), color='black', linestyle='--', linewidth=0.8, alpha=0.5)
            ax.axvline(x=log2fc_threshold, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
            ax.axvline(x=-log2fc_threshold, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
            
            # Label top genes
            # Top upregulated
            up_genes = df[(df['is_upregulated'] == 1)].head(n_label_genes // 2)
            for _, row in up_genes.iterrows():
                ax.annotate(
                    row['gene'],
                    (row['log2_fold_change'], row['neg_log10_padj']),
                    fontsize=7,
                    alpha=0.8,
                    ha='left'
                )
            
            # Top downregulated
            down_genes = df[(df['is_downregulated'] == 1)].head(n_label_genes // 2)
            for _, row in down_genes.iterrows():
                ax.annotate(
                    row['gene'],
                    (row['log2_fold_change'], row['neg_log10_padj']),
                    fontsize=7,
                    alpha=0.8,
                    ha='right'
                )
            
            # Labels and title
            ax.set_xlabel('Log2 Fold Change', fontsize=12, fontweight='bold')
            ax.set_ylabel('-Log10(Adjusted P-value)', fontsize=12, fontweight='bold')
            
            n_up = df.attrs.get('n_upregulated', 0)
            n_down = df.attrs.get('n_downregulated', 0)
            ax.set_title(
                f'{ct_name}: Differential Expression\n'
                f'(vs all other cell types, {n_up} up / {n_down} down)',
                fontsize=14, fontweight='bold'
            )
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', label=f'Upregulated (n={n_up})'),
                Patch(facecolor='blue', label=f'Downregulated (n={n_down})'),
                Patch(facecolor='lightgrey', label='Not significant')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(os.path.join(de_dir, f"volcano_{safe_name}.png"), dpi=300)
            plt.close()
            
            print(f"  - Saved volcano_{safe_name}.png")

# ==============================================================================
# SECTION 13: MAIN FUNCTION
# ==============================================================================

def load_data(path: str, is_labels: bool = False) -> Union[np.ndarray, torch.Tensor, list]:
    """
    Helper to load CSV data.
    Expects CSV format: Rows = Cells/Spots, Columns = Genes.
    First column is assumed to be the Index/Barcode.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    print(f"Loading {path}...")
    
    if is_labels:
        # Assumes a CSV where the first column is ID and second is Label, OR just a single column of labels
        df = pd.read_csv(path, index_col=0, header=None if not os.path.exists(path) else 0)
        # If the dataframe is empty or weird, try reading without index
        if df.shape[1] == 0:
            df = pd.read_csv(path, header=None)
        
        # Return the values as a flat list/array
        return df.iloc[:, 0].values.flatten()
    else:
        # Counts Matrix: Index col 0 = barcodes, Header row 0 = genes
        df = pd.read_csv(path, index_col=0)
        return df.values, df.columns.tolist(), df.index.tolist()

def main():
    parser = argparse.ArgumentParser(description="STVAE: Spatial Deconvolution Standalone")
    
    # Input Arguments
    parser.add_argument('--sc_counts', type=str, required=True, help='Path to scRNA-seq counts CSV')
    parser.add_argument('--sc_labels', type=str, required=True, help='Path to scRNA-seq labels CSV')
    parser.add_argument('--st_counts', type=str, required=True, help='Path to Spatial counts CSV')
    parser.add_argument('--st_coords', type=str, default=None, 
                        help='Path to spatial coordinates CSV')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results')
    
    # Hyperparameters
    parser.add_argument('--sc_epochs', type=int, default=200, help='scVAE training epochs')
    parser.add_argument('--st_epochs', type=int, default=2000, help='stVAE training epochs')
    parser.add_argument('--latent', type=int, default=10, help='Latent dimension size')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--gpu', action='store_true', help='Force usage of GPU if available')

    # Model Settings
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--reg_entropy', type=float, default=10.0, help='Entropy regularization')
    parser.add_argument('--reg_sparsity', type=float, default=1.0, help='Sparsity regularization')
    parser.add_argument('--temperature', type=float, default=0.5, help='Softmax temperature')
    
    # Unknown Cell Type
    parser.add_argument('--include_unknown', action='store_true', help='Add unknown cell type category')

    # Pseudo-Spot Arguments
    parser.add_argument('--use_pseudo_spots', action='store_true', help='Enable consistency training')
    parser.add_argument('--n_pseudo_spots', type=int, default=1000, help='Number of synthetic spots')
    parser.add_argument('--cells_per_spot_range', type=int, nargs=2, default=[5, 15])
    parser.add_argument('--pseudo_weight', type=float, default=1.0, help='Weight of pseudo-spot loss')
    parser.add_argument('--pseudo_warmup_epochs', type=int, default=50)
    parser.add_argument('--pseudo_training_ratio', type=int, default=1)
    
    # Visualization
    parser.add_argument('--hex_orientation', type=float, default=0.0,
                        help='Hexagon orientation in degrees (0=pointy-top, 30=flat-top)')
    parser.add_argument('--presence_threshold', type=float, default=0.05,
                        help='Minimum proportion threshold for cell type visualization (default: 0.05)')
    parser.add_argument('--latent_combination', type=str, default='pca',
                        choices=['pca', 'sum', 'norm', 'weighted', 'latent0'],
                        help='Method to combine latent dimensions for visualization: '
                             'pca (PCA first component), sum (simple sum), '
                             'norm (L2 norm), weighted (variance-weighted), '
                             'latent0 (first dimension only, original behavior)')
    # Mode Selection
    parser.add_argument('--mode', type=str, default='full', 
                        choices=['full', 'proportion_only'],
                        help='Deconvolution mode')
    parser.add_argument('--proportion_method', type=str, default='simplex_ae',
                        choices=['nnls', 'softmax_regression', 'simplex_ae'],
                        help='Method for proportion-only mode')
    parser.add_argument('--inference_mode', type=str, default='amortized',
                        choices=['amortized', 'non_amortized'],
                        help='Inference mode for gamma in full mode')
    
    # Start arguments (SIMPLIFIED)
    parser.add_argument('--two_stage', action='store_true',
                    help='Use two-stage training: stAE → STVAE')
    parser.add_argument('--freeze_proportions', action='store_true',
                        help='Fix proportions to AE values (no learning)')
    parser.add_argument('--freeze_intensity', action='store_true',
                        help='Fix intensity maps to AE values (freeze alpha)')
    parser.add_argument('--soft_constraint', action='store_true',
                        help='Use soft KL constraint on proportions instead of hard freeze')
    parser.add_argument('--constraint_strength', type=float, default=10.0,
                        help='Strength of soft constraint on proportions (KL weight)')
    parser.add_argument('--decorrelate_gamma', type=float, default=0.0,
                        help='Weight for gamma-pi decorrelation penalty')
    
    args = parser.parse_args()
    
    # Set hexagon orientation globally
    global HEXAGON_ORIENTATION
    HEXAGON_ORIENTATION = np.radians(args.hex_orientation)
    
    # 1. Setup Output Directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 2. Load Data
    print("=" * 60)
    print("Loading Data")
    print("=" * 60)
    
    # Load scRNA-seq
    X_sc, sc_genes, _ = load_data(args.sc_counts)
    cell_types_raw = load_data(args.sc_labels, is_labels=True)
    
    # Process Labels
    unique_types = np.unique(cell_types_raw)
    print(f"Found {len(unique_types)} cell types: {unique_types}")
    type_to_idx = {t: i for i, t in enumerate(unique_types)}
    cell_types = np.array([type_to_idx[t] for t in cell_types_raw])
    
    # Load Spatial Data
    X_st, st_genes, st_barcodes = load_data(args.st_counts)
    
    # --- Load Coordinates ---
    spatial_coords = None
    spatial_coords_full = None
    full_coord_barcodes = None
    matched_mask = None
    
    if args.st_coords:
        print(f"Loading coordinates from {args.st_coords}...")
        try:
            df_preview = pd.read_csv(args.st_coords, nrows=3, header=None)
            
            if df_preview.shape[1] == 6:
                print("  -> Detected raw 10x Genomics format (no header).")
                st_coords_df = pd.read_csv(args.st_coords, header=None, index_col=0)
                st_coords_df = st_coords_df.iloc[:, [4, 3]]
                st_coords_df.columns = ['x', 'y']
            else:
                print("  -> Detected standard/formatted CSV.")
                st_coords_df = pd.read_csv(args.st_coords, index_col=0)
                st_coords_df = st_coords_df.iloc[:, :2]
                st_coords_df.columns = ['x', 'y']

            # Fix Barcode Mismatches
            count_has_suffix = str(st_barcodes[0]).endswith("-1")
            coord_has_suffix = str(st_coords_df.index[0]).endswith("-1")

            if coord_has_suffix and not count_has_suffix:
                print("  -> Removing '-1' suffix from coordinate barcodes to match counts.")
                st_coords_df.index = st_coords_df.index.str.replace("-1", "", regex=False)
            elif not coord_has_suffix and count_has_suffix:
                print("  -> Adding '-1' suffix to coordinate barcodes to match counts.")
                st_coords_df.index = st_coords_df.index + "-1"

            # Store ALL coordinates for background plotting
            spatial_coords_full = st_coords_df.values.copy()
            full_coord_barcodes = st_coords_df.index.tolist()
            
            # Find which coordinate barcodes match count barcodes
            st_barcodes_set = set(st_barcodes)
            matched_mask = np.array([bc in st_barcodes_set for bc in full_coord_barcodes])
            
            n_coord_total = len(full_coord_barcodes)
            n_matched = matched_mask.sum()
            n_unmatched = n_coord_total - n_matched
            
            print(f"  -> Coordinate file has {n_coord_total} spots")
            print(f"  -> Count file has {len(st_barcodes)} spots")
            print(f"  -> Matched: {n_matched}, Unmatched (background): {n_unmatched}")
            
            if n_matched == 0:
                raise ValueError("No matching barcodes between counts and coordinates!")
            
            if n_matched < len(st_barcodes):
                print(f"  -> WARNING: {len(st_barcodes) - n_matched} count barcodes have no coordinates!")

            # Alignment: Ensure coords match the order of X_st (counts)
            st_coords_aligned = st_coords_df.reindex(st_barcodes)

            if st_coords_aligned.isnull().any().any():
                n_missing = st_coords_aligned.isnull().any(axis=1).sum()
                print(f"  -> Warning: {n_missing} spots in count matrix are missing coordinates. Filling with 0.")
                st_coords_aligned = st_coords_aligned.fillna(0)
            
            spatial_coords = st_coords_aligned.values
            print(f"  -> Matched coordinates aligned. Shape: {spatial_coords.shape}")

        except Exception as e:
            print(f"Error processing coordinates: {e}")
            raise
    else:
        print("No coordinates provided. Spatial maps will be skipped.")

    # 3. Data Intersection
    print("Intersecting genes between single-cell and spatial data...")
    common_genes = sorted(list(set(sc_genes) & set(st_genes)))
    print(f"Found {len(common_genes)} common genes.")
    
    if len(common_genes) < 50:
        raise ValueError(f"Too few common genes ({len(common_genes)}).")
        
    sc_gene_indices = [sc_genes.index(g) for g in common_genes]
    st_gene_indices = [st_genes.index(g) for g in common_genes]
    
    X_sc = X_sc[:, sc_gene_indices]
    X_st = X_st[:, st_gene_indices]
    
    print(f"Final scRNA-seq shape: {X_sc.shape}")
    print(f"Final Spatial shape:   {X_st.shape}")
    
    # 4. Configure Model
    config = STVAEConfig(
        n_genes=len(common_genes),
        n_cell_types=len(unique_types),
        n_latent=args.latent,
        sc_max_epochs=args.sc_epochs,
        st_max_epochs=args.st_epochs,
        sc_batch_size=args.batch_size,
        st_batch_size=args.batch_size,
        st_learning_rate=args.lr,
        use_pseudo_spots=args.use_pseudo_spots,
        n_pseudo_spots=args.n_pseudo_spots,
        cells_per_spot_range=args.cells_per_spot_range,
        pseudo_weight=args.pseudo_weight,
        pseudo_warmup_epochs=args.pseudo_warmup_epochs,
        pseudo_training_ratio=args.pseudo_training_ratio,
        add_unknown_cell_type=args.include_unknown,
        proportion_temperature=args.temperature,
        lambda_pi_entropy=args.reg_entropy,
        lambda_pi_sparsity=args.reg_sparsity,
        amortized_gamma=(args.inference_mode == 'amortized'),
        inference_mode=args.inference_mode,
    )
    
    # Initialize STVAE
    stvae = STVAE(config)
    
    # === MODE BRANCHING ===
    if args.mode == 'full':
        # Full VAE/AE-style with cell states
        print("\n" + "=" * 60)
        print("Running FULL mode (VAE/AE-style with cell state inference)")
        print(f"Inference Mode: {args.inference_mode.upper()}")
        print("=" * 60)
        
        # === STEP 1: FIT scVAE FIRST ===
        stvae.fit_sc(
            X=X_sc,
            cell_types=cell_types,
            gene_names=common_genes,
            cell_type_names=list(unique_types),
            n_epochs=args.sc_epochs,
            verbose=True
        )
        
        # === STEP 2: Create Pseudo-Spot Generator (ONLY for amortized mode) ===
        pseudo_gen = None
        if args.use_pseudo_spots:
            if args.inference_mode == 'amortized':
                print(f"Initializing Pseudo-Spot Generator...")
                X_sc_tensor = torch.tensor(X_sc, dtype=torch.float32).to(DEVICE)
                cell_types_tensor = torch.tensor(cell_types, dtype=torch.long).to(DEVICE)
                
                pseudo_gen = PseudoSpotGenerator(
                    X_sc=X_sc_tensor,
                    cell_types=cell_types_tensor,
                    n_cell_types=len(unique_types),
                    n_pseudo_spots=args.n_pseudo_spots,
                    cells_per_spot_range=args.cells_per_spot_range,
                    device=DEVICE
                )
            else:
                print("WARNING: Pseudo-spot training disabled for non-amortized mode.")
        
        # === STEP 3: FIT SPATIAL MODEL ===
        if args.two_stage:
            # === NEW: Run Stage 2a (stAE) and save results BEFORE stVAE ===
            print("\n" + "=" * 60)
            print("Stage 2a: Running stAE for proportions and intensity maps")
            print("=" * 60)
            
            stae_results = WarmStartTrainer.get_initial_proportions_and_intensity(
                X_st=X_st,
                X_sc=X_sc,
                cell_types=cell_types,
                n_components=config.n_latent,
                n_epochs=500,  # stAE epochs
                config=config,
                device=DEVICE
            )
            
            # === SAVE stAE (Stage 2a) RESULTS IMMEDIATELY ===
            print("\n" + "-" * 40)
            print("Saving Stage 2a (stAE) Results")
            print("-" * 40)
            
            # Save stAE proportions
            stae_proportions = stae_results['proportions']
            df_stae_props = pd.DataFrame(
                stae_proportions,
                index=st_barcodes,
                columns=list(unique_types)
            )
            df_stae_props.to_csv(os.path.join(args.output_dir, "proportions.csv"))
            print(f"Saved proportions.csv (stAE Stage 2a) - Shape: {df_stae_props.shape}")
            
            # Save stAE intensity maps (normalized)
            stae_intensity = stae_results['intensity_maps']
            df_stae_intensity = pd.DataFrame(
                stae_intensity,
                index=st_barcodes,
                columns=list(unique_types)
            )
            df_stae_intensity.to_csv(os.path.join(args.output_dir, "intensity_maps.csv"))
            print(f"Saved intensity_maps.csv (stAE Stage 2a) - Shape: {df_stae_intensity.shape}")
            
            # Save stAE intensity maps (raw, unnormalized)
            stae_intensity_raw = stae_results['intensity_maps_raw']
            df_stae_intensity_raw = pd.DataFrame(
                stae_intensity_raw,
                index=st_barcodes,
                columns=list(unique_types)
            )
            df_stae_intensity_raw.to_csv(os.path.join(args.output_dir, "intensity_maps_raw.csv"))
            print(f"Saved intensity_maps_raw.csv (stAE Stage 2a) - Shape: {df_stae_intensity_raw.shape}")
            
            # Save alpha (gene correction factors)
            stae_alpha = stae_results['alpha']
            df_alpha = pd.DataFrame(
                {'alpha': stae_alpha},
                index=common_genes
            )
            df_alpha.to_csv(os.path.join(args.output_dir, "alpha_gene_correction.csv"))
            print(f"Saved alpha_gene_correction.csv - Shape: {df_alpha.shape}")
            
            # Print stAE summary
            print("\nstAE Proportion Summary:")
            print(df_stae_props.describe())
            
            print("\nstAE Intensity Summary:")
            print(df_stae_intensity.describe())
            
            # Store for visualization (use stAE results)
            stvae.stae_results = stae_results
            
            # NOW run the full two-stage training (stVAE part)
            stvae.fit_spatial_two_stage(
                X=X_st,
                X_sc=X_sc,
                cell_types=cell_types,
                coords=spatial_coords,
                n_epochs=args.st_epochs,
                verbose=True,
                freeze_proportions=args.freeze_proportions,
                freeze_intensity=args.freeze_intensity,
                soft_constraint=args.soft_constraint,
                constraint_strength=args.constraint_strength,
                decorrelate_gamma=args.decorrelate_gamma
            )
            
            # === NEW: Save stVAE-refined proportions and intensity maps ===
            print("\n" + "-" * 40)
            print("Saving Stage 2b (stVAE-refined) Results")
            print("-" * 40)
            
            # Get stVAE-refined proportions
            stvae_proportions = stvae.get_proportions(include_unknown=False)
            df_stvae_props = pd.DataFrame(
                stvae_proportions,
                index=st_barcodes,
                columns=list(unique_types)
            )
            df_stvae_props.to_csv(os.path.join(args.output_dir, "proportions_stvae_refined.csv"))
            print(f"Saved proportions_stvae_refined.csv (stVAE Stage 2b) - Shape: {df_stvae_props.shape}")
            
            # Compute stVAE-refined intensity maps
            # Get alpha from stVAE model
            stvae_alpha = stvae.st_model.alpha.detach().cpu().numpy()
            
            # Get reference profiles (same as stAE used)
            ref_profiles = stae_results['reference_profiles']  # [n_cell_types, n_genes]
            
            # Compute intensity: library_size × π × (α · profile).sum()
            library_sizes = X_st.sum(axis=1, keepdims=True) if isinstance(X_st, np.ndarray) else X_st.sum(dim=1, keepdim=True).cpu().numpy()
            profile_sums = (stvae_alpha[np.newaxis, :] * ref_profiles).sum(axis=-1)  # [n_cell_types]
            stvae_intensity_raw = library_sizes * stvae_proportions * profile_sums[np.newaxis, :]
            
            # Normalize intensity maps
            intensity_max = stvae_intensity_raw.max(axis=0, keepdims=True)
            intensity_max[intensity_max == 0] = 1.0
            stvae_intensity = stvae_intensity_raw / intensity_max
            
            df_stvae_intensity = pd.DataFrame(
                stvae_intensity,
                index=st_barcodes,
                columns=list(unique_types)
            )
            df_stvae_intensity.to_csv(os.path.join(args.output_dir, "intensity_maps_stvae_refined.csv"))
            print(f"Saved intensity_maps_stvae_refined.csv (stVAE Stage 2b) - Shape: {df_stvae_intensity.shape}")
            
            # Save raw intensity
            df_stvae_intensity_raw = pd.DataFrame(
                stvae_intensity_raw,
                index=st_barcodes,
                columns=list(unique_types)
            )
            df_stvae_intensity_raw.to_csv(os.path.join(args.output_dir, "intensity_maps_raw_stvae_refined.csv"))
            print(f"Saved intensity_maps_raw_stvae_refined.csv - Shape: {df_stvae_intensity_raw.shape}")
            
            # Store stVAE results for visualization
            stvae.stvae_results = {
                'proportions': stvae_proportions,
                'intensity_maps': stvae_intensity,
                'intensity_maps_raw': stvae_intensity_raw,
                'alpha': stvae_alpha
            }
        else:
            stvae.fit_spatial(
                X=X_st,
                n_epochs=args.st_epochs,
                pseudo_generator=pseudo_gen,
                verbose=True
            )
    
    # ==========================================================================
    # SAVE RESULTS (Both modes)
    # ==========================================================================
    print("\n" + "=" * 60)
    print(f"Saving Results to {args.output_dir}")
    print("=" * 60)
    
    # Save model (if not NNLS)
    if args.mode == 'full' or args.proportion_method != 'nnls':
        stvae.save(os.path.join(args.output_dir, "stvae_model.pt"))
        print("Saved stvae_model.pt")
    
    # === MODIFIED: Only save VAE proportions if NOT using two_stage ===
    # (two_stage already saved stAE proportions above)
    if args.mode == 'full' and args.two_stage:
        # Use stAE proportions (already saved above)
        # Just create df_props reference for visualization
        df_props = pd.DataFrame(
            stvae.stae_results['proportions'],
            index=st_barcodes,
            columns=list(unique_types)
        )
        print("Using stAE proportions from Stage 2a (already saved)")
    else:
        # Get and save proportions (standard mode or proportion_only)
        proportions = stvae.get_proportions(include_unknown=False)
        
        columns = list(unique_types)
        df_props = pd.DataFrame(
            proportions,
            index=st_barcodes,
            columns=columns
        )
        df_props.to_csv(os.path.join(args.output_dir, "proportions.csv"))
        print(f"Saved proportions.csv (Shape: {df_props.shape})")
        
        # Print proportion summary
        print("\nProportion Summary:")
        print(df_props.describe())
    
    # ==========================================================================
    # SAVE GAMMA STATES AND IMPUTED EXPRESSION (FULL MODE ONLY)
    # ==========================================================================
    if args.mode == 'full':
        print("\n" + "=" * 60)
        print("Saving Cell States and Imputed Expression (Full Mode)")
        print("=" * 60)
        
        # Save raw latent states (Gamma)
        print("\nSaving raw latent states (Gamma)...")
        gamma_states = stvae.get_cell_type_states(X_st, return_numpy=True)
        np.save(os.path.join(args.output_dir, "gamma_states.npy"), gamma_states)
        print(f"Saved gamma_states.npy (Shape: {gamma_states.shape})")

        # Save flattened CSV for each cell type
        for i, ct_name in enumerate(unique_types):
            safe_name = str(ct_name).replace("/", "_").replace(" ", "_")
            ct_states = gamma_states[:, i, :]
            cols = [f"Latent_{k}" for k in range(config.n_latent)]
            df_gamma = pd.DataFrame(ct_states, index=st_barcodes, columns=cols)
            df_gamma.to_csv(os.path.join(args.output_dir, f"gamma_states_{safe_name}.csv"))
            print(f"Saved gamma_states_{safe_name}.csv")

        # Save imputed expression
        print("\nSaving imputed expression...")
        for i, ct_name in enumerate(unique_types):
            safe_name = str(ct_name).replace("/", "_").replace(" ", "_")
            imputed = stvae.impute_expression(X_st, cell_type=i)
            df_imputed = pd.DataFrame(imputed, index=st_barcodes, columns=common_genes)
            df_imputed.to_csv(os.path.join(args.output_dir, f"imputed_expr_{safe_name}.csv"))
            print(f"Saved imputed_expr_{safe_name}.csv")
    
    # =====================================================================
    # ADD THIS: Save Denoised Tissue Reconstruction
    # =====================================================================
    print("\nSaving Denoised Tissue Reconstruction...")
    # Mathematically reconstruct the whole tissue: library_size * alpha * sum(pi * rho)
    stvae.st_model.eval()
    with torch.no_grad():
        X_st_tensor = torch.tensor(X_st, dtype=torch.float32).to(DEVICE)
        spot_indices = torch.arange(X_st_tensor.shape[0]).to(DEVICE)
        
        # Run forward pass to get 'mu' (the denoised expected counts)
        outputs = stvae.st_model(
            x=X_st_tensor, 
            spot_idx=spot_indices, 
            proportions_module=stvae.proportions
        )
        denoised_tissue = outputs['mu'].cpu().numpy()
        
    df_denoised = pd.DataFrame(denoised_tissue, index=st_barcodes, columns=common_genes)
    df_denoised.to_csv(os.path.join(args.output_dir, "denoised_tissue_reconstruction.csv"))
    print(f"Saved denoised_tissue_reconstruction.csv (Shape: {df_denoised.shape})")

    # ==========================================================================
    # VISUALIZATION GENERATION
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    
    # === Use stAE proportions for visualization when two_stage ===
    if args.mode == 'full' and args.two_stage and hasattr(stvae, 'stae_results'):
        df_props_for_vis = pd.DataFrame(
            stvae.stae_results['proportions'],
            index=st_barcodes,
            columns=list(unique_types)
        )
        print("Using stAE (Stage 2a) proportions for visualizations")
    else:
        df_props_for_vis = df_props
    
    # 1. Co-occurrence Heatmap (Both modes)
    VisualizationUtils.plot_cooccurrence(
        df_props_for_vis,
        os.path.join(args.output_dir, "cooccurrence_heatmap.png")
    )
    print("- Saved cooccurrence_heatmap.png")
    
    # 2. Spatial Maps (Both modes, if coords provided)
    if spatial_coords is not None:
        print("\nGenerating spatial maps...")
        print(f"Using presence threshold: {args.presence_threshold:.1%}")
        
        # 2a. stAE spatial maps (original, unmodified by stVAE)
        VisualizationUtils.plot_spatial_maps(
            df_props_for_vis,
            spatial_coords,
            args.output_dir,
            coords_full=spatial_coords_full,
            matched_mask=matched_mask,
            presence_threshold=args.presence_threshold
        )
        print("- Saved spatial_proportion_maps.png (stAE Stage 2a)")
        print("- Saved spatial_dominant_type.png (stAE Stage 2a)")
        
        # 2b. stVAE-refined spatial maps (if two_stage mode with soft constraint)
        if args.mode == 'full' and args.two_stage and hasattr(stvae, 'stvae_results'):
            print("\nGenerating stVAE-refined spatial maps...")
            
            df_props_stvae = pd.DataFrame(
                stvae.stvae_results['proportions'],
                index=st_barcodes,
                columns=list(unique_types)
            )
            
            # Create subdirectory for stVAE-refined outputs
            stvae_viz_dir = os.path.join(args.output_dir, "stvae_refined")
            os.makedirs(stvae_viz_dir, exist_ok=True)
            
            VisualizationUtils.plot_spatial_maps(
                df_props_stvae,
                spatial_coords,
                stvae_viz_dir,
                coords_full=spatial_coords_full,
                matched_mask=matched_mask,
                presence_threshold=args.presence_threshold
            )
            print("- Saved stvae_refined/spatial_proportion_maps.png (stVAE Stage 2b)")
            print("- Saved stvae_refined/spatial_dominant_type.png (stVAE Stage 2b)")

        # 2c. Marker gene spatial maps
        print("\nGenerating marker gene visualizations...")
        
        # Get reference profiles
        if args.mode == 'full' and args.two_stage and hasattr(stvae, 'stae_results'):
            ref_profiles = stvae.stae_results['reference_profiles']
        else:
            # Compute reference profiles if not available
            X_sc_tensor = torch.from_numpy(X_sc).float()
            cell_types_tensor = torch.from_numpy(cell_types).long()
            ref_profiles_module = ReferenceProfiles(
                X_sc_tensor, cell_types_tensor, len(unique_types), normalize='cpm'
            )
            ref_profiles = ref_profiles_module.profiles.numpy()
        
        # Generate marker gene spatial maps for each cell type
        VisualizationUtils.plot_cell_type_marker_genes(
            df_props=df_props_for_vis,
            X_st=X_st if isinstance(X_st, np.ndarray) else X_st.numpy(),
            gene_names=common_genes,
            reference_profiles=ref_profiles,
            coords=spatial_coords,
            output_dir=args.output_dir,
            n_top_genes=5,
            coords_full=spatial_coords_full,
            matched_mask=matched_mask,
            presence_threshold=args.presence_threshold
        )
        
        # Generate dot plot
        VisualizationUtils.plot_marker_gene_dotplot(
            X_sc=X_sc,
            cell_types=cell_types,
            gene_names=common_genes,
            cell_type_names=list(unique_types),
            reference_profiles=ref_profiles,
            output_path=os.path.join(args.output_dir, "marker_genes_dotplot.png"),
            n_top_genes=5,
            expression_threshold=0.5
        )
        print("- Saved marker gene visualizations")
    else:
        print("\nSkipping spatial maps (no coordinates provided)")
    
    # 3. Latent Space Visualizations (FULL MODE ONLY)
    if args.mode == 'full':
        print("\nGenerating latent space visualizations...")
        
        # Get gamma states
        gamma = stvae.get_cell_type_states(X_st, return_numpy=True)
        spot_latent = gamma.reshape(gamma.shape[0], -1)
        
        # Main latent UMAP
        VisualizationUtils.plot_latent_projection(
            spot_latent,
            df_props,
            os.path.join(args.output_dir, "latent_umap.png")
        )
        print("- Saved latent_umap.png")
        
        # Cell-type-specific continuum visualizations
        print("\nGenerating cell-type-specific state visualizations...")
        print(f"Using presence threshold: {args.presence_threshold:.1%}")
        print(f"Using latent combination method: {args.latent_combination}")
        for i, ct_name in enumerate(unique_types):
            VisualizationUtils.plot_cell_type_continuum(
                gamma_states=gamma,
                df_props=df_props,
                cell_type_idx=i,
                cell_type_name=ct_name,
                coords=spatial_coords,
                output_dir=args.output_dir,
                coords_full=spatial_coords_full,
                matched_mask=matched_mask,
                presence_threshold=args.presence_threshold,
                latent_combination_method=args.latent_combination  # NEW ARGUMENT
            )
        
        # Full UMAP visualizations with consistent layout across all cell types
        print("\nGenerating consistent full-spot UMAP visualizations...")
        print(f"Using latent combination method: {args.latent_combination}")
        global_umap_embedding = None
        for i, ct_name in enumerate(unique_types):
            global_umap_embedding = VisualizationUtils.plot_cell_type_continuum_full(
                gamma_states=gamma,
                df_props=df_props,
                cell_type_idx=i,
                cell_type_name=ct_name,
                output_dir=args.output_dir,
                proportion_threshold=args.presence_threshold,
                global_embedding=global_umap_embedding,
                latent_combination_method=args.latent_combination  # NEW ARGUMENT
            )
        
        # Save the global embedding for potential downstream use
        if global_umap_embedding is not None:
            np.save(os.path.join(args.output_dir, "global_umap_embedding.npy"), global_umap_embedding)
            print(f"- Saved global_umap_embedding.npy (Shape: {global_umap_embedding.shape})")
    
    else:
        print("\nNote: Latent space visualizations skipped in proportion_only mode.")
    
    # ==========================================================================
    # GENE RANKING AND DIFFERENTIAL EXPRESSION ANALYSIS (BOTH MODES)
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Computing Gene Rankings and Differential Expression")
    print("=" * 60)
    
    # Get reference profiles (compute if not already available)
    if args.mode == 'full' and args.two_stage and hasattr(stvae, 'stae_results'):
        ref_profiles = stvae.stae_results['reference_profiles']
    else:
        # Compute reference profiles
        X_sc_tensor = torch.from_numpy(X_sc).float()
        cell_types_tensor = torch.from_numpy(cell_types).long()
        ref_profiles_module = ReferenceProfiles(
            X_sc_tensor, cell_types_tensor, len(unique_types), normalize='cpm'
        )
        ref_profiles = ref_profiles_module.profiles.numpy()
    
    # 1. Compute marker gene rankings from reference profiles
    print("\nComputing marker gene rankings...")
    marker_results = MarkerGeneAnalysis.compute_marker_scores(
        reference_profiles=ref_profiles,
        gene_names=common_genes,
        cell_type_names=list(unique_types)
    )
    
    # 2. Compute differential expression (one vs rest)
    print("\nComputing differential expression analysis...")
    de_results = MarkerGeneAnalysis.compute_differential_expression(
        X_sc=X_sc,
        cell_types=cell_types,
        gene_names=common_genes,
        cell_type_names=list(unique_types),
        min_cells=10,
        min_pct_expressing=0.1
    )
    
    # 3. Save results
    MarkerGeneAnalysis.save_results(
        marker_results=marker_results,
        de_results=de_results,
        output_dir=args.output_dir
    )
    
    # 4. Generate volcano plots
    MarkerGeneAnalysis.plot_volcano_plots(
        de_results=de_results,
        output_dir=args.output_dir,
        log2fc_threshold=0.5,
        padj_threshold=0.05,
        n_label_genes=10
    )

    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    print("\n" + "=" * 60)
    print("STVAE Pipeline Finished Successfully")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Presence threshold for visualization: {args.presence_threshold:.1%}")
    if args.mode == 'proportion_only':
        print(f"Method: {args.proportion_method}")
    if args.mode == 'full':
        print(f"Inference Mode: {args.inference_mode}")
        if args.two_stage:
            print(f"\nTwo-stage training: Enabled")
            print(f"  Stage 2a (stAE): Proportion & intensity estimation with fixed profiles")
            print(f"  Stage 2b (stVAE): Cell state (γ) inference with frozen scVAE decoder")
            print(f"\n  Training constraints:")
            if args.soft_constraint:
                print(f"    - Soft constraint on π: ENABLED (strength={args.constraint_strength})")
                print(f"    - Decorrelation penalty: {args.decorrelate_gamma}")
            else:
                print(f"    - Freeze proportions (π): {args.freeze_proportions}")
            print(f"    - Freeze gene correction (α): {args.freeze_intensity}")
            print(f"\n  Note on α (gene correction factor):")
            print(f"    α captures gene-specific technical effects between platforms.")
            print(f"    Intensity maps are DERIVED: intensity = lib_size × π × (α · profile).sum()")
            print(f"\n  Final outputs use stAE (Stage 2a) results for proportions/intensity.")
    
    print(f"\nOutput directory: {args.output_dir}")
    print("\nGenerated files:")
    
    # List generated files
    generated_files = []
    generated_files.append("  - proportions.csv (from stAE Stage 2a)" if (args.mode == 'full' and args.two_stage) else "  - proportions.csv")
    
    if args.mode == 'full' and args.two_stage:
        generated_files.append("  - intensity_maps.csv (from stAE Stage 2a)")
        generated_files.append("  - intensity_maps_raw.csv (from stAE Stage 2a)")
        generated_files.append("  - alpha_gene_correction.csv (from stAE Stage 2a)")
        generated_files.append("  - proportions_stvae_refined.csv (from stVAE Stage 2b)")
        generated_files.append("  - intensity_maps_stvae_refined.csv (from stVAE Stage 2b)")
        generated_files.append("  - intensity_maps_raw_stvae_refined.csv (from stVAE Stage 2b)")
        generated_files.append("  - gene_analysis_summary.txt")
        generated_files.append("  - marker_gene_rankings/")
        for ct_name in unique_types:
            safe_name = str(ct_name).replace("/", "_").replace(" ", "_").replace(".", "_")
            generated_files.append(f"      - {safe_name}_markers.csv")
        generated_files.append("  - differential_expression/")
        for ct_name in unique_types:
            safe_name = str(ct_name).replace("/", "_").replace(" ", "_").replace(".", "_")
            generated_files.append(f"      - {safe_name}_de.csv")
            generated_files.append(f"      - volcano_{safe_name}.png")
        if spatial_coords is not None:
            generated_files.append("  - stvae_refined/spatial_proportion_maps.png (stVAE Stage 2b)")
            generated_files.append("  - stvae_refined/spatial_dominant_type.png (stVAE Stage 2b)")
    
    if args.mode == 'full' or args.proportion_method != 'nnls':
        generated_files.append("  - stvae_model.pt")
    if args.mode == 'full':
        generated_files.append("  - gamma_states.npy")
        for ct_name in unique_types:
            safe_name = str(ct_name).replace("/", "_").replace(" ", "_")
            generated_files.append(f"  - gamma_states_{safe_name}.csv")
            generated_files.append(f"  - imputed_expr_{safe_name}.csv")
    generated_files.append("  - cooccurrence_heatmap.png")
    
    if spatial_coords is not None:
        generated_files.append("  - spatial_proportion_maps.png")
        generated_files.append("  - spatial_dominant_type.png")
        # Marker gene files
        for ct_name in unique_types:
            safe_name = str(ct_name).replace("/", "_").replace(" ", "_")
            generated_files.append(f"  - marker_genes_{safe_name}.png")
        generated_files.append("  - marker_genes_summary.png")
        generated_files.append("  - marker_genes_dotplot.png")
    
    if args.mode == 'full':
        generated_files.append("  - latent_umap.png")
        for ct_name in unique_types:
            safe_name = str(ct_name).replace("/", "_").replace(" ", "_")
            generated_files.append(f"  - continuum_{safe_name}.png")
            if spatial_coords is not None:
                generated_files.append(f"  - spatial_state_{safe_name}.png")
            generated_files.append(f"  - continuum_full_{safe_name}.png")
        generated_files.append("  - global_umap_embedding.npy")
    
    for f in generated_files:
        print(f)

if __name__ == "__main__":
    main()