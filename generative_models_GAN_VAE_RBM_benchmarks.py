#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 19:11:58 2026

@author: Marcin Plodzien
course: Machine Learning for Quantum and Classical Physics 2025/2026

Classical Generative Benchmarks: Parameter-Constrained Comparison
=================================================================
Purpose:
    Comparison between standard classical 
    generative models (GAN, VAE, RBM).
   

Models Implemented:
    1. GAN (Generative Adversarial Network): Adversarial Minimax game.
    2. VAE (Variational Autoencoder): Probabilistic compression via ELBO.
    3. RBM (Restricted Boltzmann Machine): Energy-based model via Contrastive Divergence.

DEPENDENCIES:
    - JAX (JIT compilation, Auto-differentiation)
    - Optax (Optimization)
    - Matplotlib, Numpy, Scipy (Visualization & Stats)
================================================================================
"""

import os
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad, value_and_grad, vmap
from functools import partial
from tqdm import tqdm
import optax
import optax
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, wasserstein_distance



# ==============================================================================
#                               CONFIGURATION
# ==============================================================================
SEED = 42
N_SAMPLES_TRAIN = 100000 

# --- Training Hyperparameters ---
BATCH_SIZE = 128
N_EPOCHS = 100000

# --- Learning Rates (Tuned for Stability) ---
LR_GAN = 0.0002 # Standard stable rate for GANs
LR_VAE = 0.001 
LR_RBM = 0.01   # Energy models often prefer higher rates/SGD

# --- Evaluation Config ---
KLD_CHECK_EVERY = 500       # Interval for computing KLD metric
SNAPSHOT_PERCENT = 25        # Save plot every N% of training (e.g. 5% = 20 frames)

# --- Parameter Budget (The Equalizer) ---
PARAMETER_BUDGET = 310

OUT_DIR = "results_classical_benchmarks"
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# DISTRIBUTION CONFIGURATION
# ------------------------------------------------------------------------------
# ==============================================================================
# CONFIGURATION
# ==============================================================================
# "2D_MIX_GAUSS", "1D_BIMODAL", "1D_TRIMODAL", "1D_QUADMODAL", "1D_BETA", "1D_POISSON", "2D_GMM"
TARGET_TYPE = "2D_GMM"

# PARAMETER BUDGET (Approximate)
PARAMETER_BUDGET = 310

# DISCRETIZATION CONFIG
NUM_BINS = 20  # Set to e.g. 50 to enable discretization. 0 = Continuous.

# TARGET CORRELATION (Legacy placeholder or for simple 2D_MIX_GAUSS)
TARGET_CORRELATION = 0.0 

# GMM CONFIGURATION (For TARGET_TYPE="2D_GMM")
# List of (Weight, Mean, Covariance) tuples
if TARGET_TYPE == "2D_GMM":
    DATA_DIM = 2
    NUM_BINS = 0
    SIGMA = 0.5 # Dummy value for logging, actual covariance is in GMM_COMPONENTS
    # Weights must sum to 1.0
    # Means are [x, y]
    # Covs are [[xx, xy], [yx, yy]]
    GMM_COMPONENTS = [
        # Component 1: Correlated (Positive)
        (0.25, jnp.array([1.5, 1.5]), jnp.array([[0.5, 0.4], [0.4, 0.5]])),
        # Component 2: Correlated (Negative)
        (0.25, jnp.array([1.5, -1.5]), jnp.array([[0.5, -0.4], [-0.4, 0.5]])),
        # Component 3: Uncorrelated
        (0.25, jnp.array([-1.5, 1.5]), jnp.array([[0.5, 0.0], [0.0, 0.5]])),
        # Component 4: Different Variance
        (0.25, jnp.array([-1.5, -1.5]), jnp.array([[0.2, 0.0], [0.0, 0.8]]))
    ]
    print(f"Target Type: 2D_GMM | Components: {len(GMM_COMPONENTS)}")

elif TARGET_TYPE == "2D_MIX_GAUSS":
    # A standard 2D benchmark: 4 Gaussian modes at the corners of a square.
    DATA_DIM = 2
    MEANS = jnp.array([[1.5, 1.5], [1.5, -1.5], [-1.5, 1.5], [-1.5, -1.5]])
    SIGMA = 0.5 
    NUM_BINS = 0 # 2D is strictly continuous in this benchmark
    print(f"Target Type: 2D_MIX_GAUSS | Correlation: {TARGET_CORRELATION}")

elif TARGET_TYPE == "1D_BIMODAL":
    # Two separated clusters. Good for testing if model can capture multimodal support.
    DATA_DIM = 1
    MEANS = jnp.array([[-1.5], [1.5]])
    SIGMA = 0.4

elif TARGET_TYPE == "1D_TRIMODAL":
    # Three modes. Tests finer resolution.
    DATA_DIM = 1
    MEANS = jnp.array([[-2.0], [0.0], [2.0]])
    SIGMA = 0.35 

elif TARGET_TYPE == "1D_QUADMODAL":
    # High Frequency. Hard for low-capacity models (Mode Collapse risk).
    DATA_DIM = 1
    MEANS = jnp.array([[-2.5], [-0.8], [0.8], [2.5]])
    SIGMA = 0.3 

elif TARGET_TYPE == "1D_BETA":
    # Skewed Distribution. Tests ability to learn asymmetry (unlike Gaussian).
    DATA_DIM = 1
    BETA_A = 2.0
    BETA_B = 5.0

elif TARGET_TYPE == "1D_POISSON":
    # Discrete Count Data. 
    # NOTE: Inherently discrete, but we shift it to be centered at 0.
    DATA_DIM = 1
    POISSON_LAMBDA = 4.0
    
else:
    raise ValueError(f"Unknown TARGET_TYPE: {TARGET_TYPE}")

# Force reset Bins for 2D because 2D Logic in this script assumes continuous
if "2D" in TARGET_TYPE:
    NUM_BINS = 0
    print(f"2D Target Type detected. NUM_BINS forced to 0 (continuous).")

if NUM_BINS > 0:
    DATA_DIM = NUM_BINS
    print(f"Discretization Active: DATA_DIM adjusted to {NUM_BINS} bins.")

 
# ==============================================================================
# DATA GENERATION
# ==============================================================================
@partial(jit, static_argnums=(1,))
def sample_target(key, batch_size):
    """
    Generates synthetic training samples from the ground truth distribution P(x).
    Supports Discretization if NUM_BINS > 0.
    Supports 2D Correlation if TARGET_CORRELATION != 0 (for 2D_MIX_GAUSS).
    """
    key, subkey = jax.random.split(key)
    
    if TARGET_TYPE == "2D_GMM":
        # 1. Select Component
        weights = jnp.array([c[0] for c in GMM_COMPONENTS])
        indices = jax.random.categorical(subkey, jnp.log(weights), shape=(batch_size,))
        
        # 2. Prepare Sample (Vectorized)
        # We need to sample from N(Mean_i, Cov_i).
        # Strategy: Sample standard normal, transform by Cholesky of Cov_i, add Mean_i.
        
        # Precompute Cholesky for all components
        covs = jnp.stack([c[2] for c in GMM_COMPONENTS])
        Ls = jnp.linalg.cholesky(covs + 1e-6*jnp.eye(2)) # Add jitter for stability
        means = jnp.stack([c[1] for c in GMM_COMPONENTS])
        
        # Gather selected L and Mean
        # indices shape: (batch_size,)
        selected_L = Ls[indices]       # (batch, 2, 2)
        selected_mean = means[indices] # (batch, 2)
        
        # Sample standard noise
        noise = jax.random.normal(key, (batch_size, 2)) # (batch, 2)
        
        # Apply transform: L @ noise + mean
        # Einsum: (batch, 2, 2) * (batch, 2) -> (batch, 2)
        # x = L @ z
        # batch_dim i, out_dim j, in_dim k
        transformed_noise = jnp.einsum('ijk,ik->ij', selected_L, noise)
        
        batch = selected_mean + transformed_noise
        return batch

    elif TARGET_TYPE == "2D_MIX_GAUSS":
        # 1. Select Mode
        indices = jax.random.randint(subkey, (batch_size,), 0, 4)
        selected_means = MEANS[indices]
        # 2. Sample Gaussian Noise (Uncorrelated)
        noise = jax.random.normal(key, (batch_size, 2))
        # 3. Add Mean
        batch = selected_means + SIGMA * noise
        return batch
    elif "1D" in TARGET_TYPE and "POISSON" not in TARGET_TYPE and "BETA" not in TARGET_TYPE:
        # Standard GMM Logic for 1D
        n_modes = len(MEANS)
        indices = jax.random.randint(subkey, (batch_size,), 0, n_modes)
        selected_means = MEANS[indices]
        noise = jax.random.normal(key, (batch_size, 1))
        batch = selected_means + SIGMA * noise
        
    elif "BETA" in TARGET_TYPE:
        # Beta Distribution
        batch = jax.random.beta(key, BETA_A, BETA_B, shape=(batch_size, 1))
        batch = (batch - 0.5) * 5.0 # Scale to [-2.5, 2.5]
        
    elif "POISSON" in TARGET_TYPE:
        # Poisson
        batch = jax.random.poisson(key, POISSON_LAMBDA, shape=(batch_size, 1)).astype(jnp.float32)
        batch = batch - POISSON_LAMBDA # Center at 0
        
    else:
        # Fallback (should be covered)
        batch = jax.random.normal(key, (batch_size, DATA_DIM))

    # Apply Discretization if needed
    if NUM_BINS > 0:
        # Shape: (Batch, NUM_BINS)
        return jax.nn.one_hot(indices[:, 0], NUM_BINS)
    
    return raw

# ==============================================================================
# METRICS (KLD)
# ==============================================================================
# ==============================================================================
#                               HELPER: METRICS
# ==============================================================================
def compute_kld_histogram(samples_p, samples_q, bins=30, range_limit=None, epsilon=1e-10):
    """
    Computes Discrete KL Divergence KL(P || Q) using histograms.
    """
    d = samples_p.shape[1]
    if range_limit is None:
        if d == 2: range_limit = [[-3.5, 3.5], [-3.5, 3.5]]
        else: range_limit = [-3.5, 3.5]

    # Safety: Remove NaNs
    samples_p = samples_p[~np.isnan(samples_p).any(axis=1)]
    samples_q = samples_q[~np.isnan(samples_q).any(axis=1)]
    
    if len(samples_p) == 0 or len(samples_q) == 0:
        return np.nan

    if d == 2:
        hist_p, _, _ = np.histogram2d(samples_p[:,0], samples_p[:,1], bins=bins, range=range_limit, density=True)
        hist_q, _, _ = np.histogram2d(samples_q[:,0], samples_q[:,1], bins=bins, range=range_limit, density=True)
    else:
        hist_p, _ = np.histogram(samples_p[:,0], bins=bins, range=range_limit, density=True)
        hist_q, _ = np.histogram(samples_q[:,0], bins=bins, range=range_limit, density=True)
    
    # Avoid Division by Zero if all samples out of bounds
    sum_p = np.sum(hist_p)
    sum_q = np.sum(hist_q)
    
    if sum_p == 0 or sum_q == 0:
        return np.inf

    pdf_p = (hist_p / sum_p) + epsilon
    pdf_q = (hist_q / sum_q) + epsilon
    
    return np.sum(pdf_p * np.log(pdf_p / pdf_q))

def compute_wasserstein_1d(u_samples, v_samples):
    """
    Computes the 1D Wasserstein Distance (Earth Mover's Distance).
    
    THEORY:
    -------
    WD(P, Q) = Integral | CDF_P(x) - CDF_Q(x) | dx
    For finite samples, this is efficiently computed as the L1 distance between
    the sorted samples (quantiles).
    
    Why this matters:
    Unlike KLD, Wasserstein provides a meaningful gradient even when distributions 
    are disjoint (don't overlap). KLD would be infinite. WD is smooth using geometry.
    """
    wd = wasserstein_distance(u_samples.flatten(), v_samples.flatten())
    return wd

# ==============================================================================
#                      HELPER: DIFFERENTIABLE DISCRETIZATION
# ==============================================================================
def gumbel_softmax(logits, key, temperature=1.0, hard=False):
    """
    The Gumbel-Softmax Trick (Jang et al., 2016).
    Allows backpropagation through categorical sampling.
    
    Formula: y = Softmax( (logits + gumbel_noise) / temperature )
    
    Args:
        logits: Unnormalized log-probabilities.
        key: JAX PRNG key.
        temperature: Controls smoothness. T -> 0 approaches discrete argmax.
        hard: If True, returns One-Hot vector (forward pass) but gradients flow 
              as if it were Softmax (Straight-Through Estimator).
    """
    # 1. Sample Gumbel Noise: g = -log(-log(u)), u ~ Uniform(0,1)
    u = jax.random.uniform(key, shape=logits.shape)
    g = -jnp.log(-jnp.log(u + 1e-20) + 1e-20)
    
    # 2. Softmax with Temperature
    y = jax.nn.softmax((logits + g) / temperature)
    
    if hard:
        # Straight-Through Estimator:
        # Forward: One-Hot(Argmax)
        # Backward: Gradient of Softmax
        y_hard = jax.nn.one_hot(jnp.argmax(y, axis=-1), logits.shape[-1])
        # Return y_hard but detach y_hard - y so gradient flows through y
        return y_hard - jax.lax.stop_gradient(y) + y
    else:
        return y

# ==============================================================================
# HELPER: DECODING (BINS -> 1D CONTINUOUS)
# ==============================================================================
def decode_to_1d(samples):
    """
    Decodes Multi-dimensional (Binned) data back to 1D Continuous estimation.
    """
    if samples.shape[1] == 1: 
        return samples
    if samples.shape[1] == 2 and NUM_BINS == 0: 
        return samples
        
    # Discretized/One-Hot -> Bin Centers
    indices = jnp.argmax(samples, axis=1)
    bin_centers = np.linspace(-3.5, 3.5, samples.shape[1])
    decoded = bin_centers[indices].reshape(-1, 1)
    return decoded

# ==============================================================================
# HELPER: MLP (Multi-Layer Perceptron)
# ==============================================================================
def init_mlp(key, layers):
    """
    Initializes weights and biases for a standard MLP.
    Uses 'He Initialization' (scaling by sqrt(2/fan_in+fan_out)) for better convergence.
    
    Args:
        layers: List of integers, e.g. [16, 64, 2] defines input->hidden->output sizes.
    """
    params = []
    keys = jax.random.split(key, len(layers)-1)
    for i, (din, dout) in enumerate(zip(layers[:-1], layers[1:])):
        k_w, k_b = jax.random.split(keys[i])
        scale = jnp.sqrt(2.0/(din+dout))
        w = jax.random.normal(k_w, (din, dout)) * scale
        b = jnp.zeros(dout)
        params.append({"w": w, "b": b})
    return params

def forward_mlp(params, x, activation=jax.nn.leaky_relu, final_act=None):
    """
    Standard Feed-Forward pass.
    
    Args:
        params: List of dicts [{"w":.., "b":..}, ...]
        x: Input batch (batch, dim)
        activation: Hidden layer activation (default Leaky ReLU)
        final_act: Optional output activation (e.g. Sigmoid/Tanh)
    """
    for l in params[:-1]:
        x = activation(x @ l["w"] + l["b"])
    final = params[-1]
    x = x @ final["w"] + final["b"]
    if final_act: x = final_act(x)
    return x

# ==============================================================================
#                               METHOD 1: GENERATIVE ADVERSARIAL NETWORK (GAN)
# ==============================================================================
"""
GAN THEORETICAL FRAMEWORK
=========================

1. Concept
----------
A GAN consists of two neural networks competing in a zero-sum game:
- **Generator G(z)**: Maps latent noise z ~ N(0, I) to data space x_fake.
- **Discriminator D(x)**: Outputs probability P(x is Real).

2. The Minimax Objective
------------------------
Mathematically, they play a minimax game on the Value function V(G, D):
    min_G max_D V(D, G) = E_{x~Data}[log D(x)] + E_{z~Noise}[log(1 - D(G(z)))]

3. Loss Functions Implemented
-----------------------------

(A) Discriminator Loss (Standard Binary Cross Entropy):
    Max log D(x_real) + log(1 - D(x_fake))
    => Min -[ log D(x_real) + log(1 - D(x_fake)) ]
    
    *Implementation Detail*: We use Label Smoothing (Real=0.9 instead of 1.0).
    This prevents the Discriminator from becoming over-confident (logits -> infinity),
    which would kill gradients for the Generator.

(B) Generator Loss (Non-Saturating):
    Instead of minimizing log(1 - D(G(z))), which has vanishing gradients when
    D is very strong (D(G(z)) ~ 0), we **maximize log D(G(z))**.
    
    Heuristic: "G wants to fool D into thinking fake data is real."
    => Min -log D(G(z))

4. Architecture & Constraints
-----------------------------
- Inputs: 16-dim Gaussian Noise.
- Constraints: Hidden Dimension = 16 to keep total parameters ~300.
"""

def sample_gan(params, key, n_samples):
    """
    Inference Step for GAN:
    Draws latent noise z and passes it through the trained Generator.
    If Discrete (NUM_BINS>0), performs Argmax to get hard bins.
    """
    z = jax.random.normal(key, (n_samples, 16))
    gen_out = forward_mlp(params, z)
    
    if NUM_BINS > 0:
        # For Inference/Plotting, we want the hard categories (One-Hot)
        # We can use argmax to select the bin.
        indices = jnp.argmax(gen_out, axis=1)
        return jax.nn.one_hot(indices, NUM_BINS)
    else:
        return np.array(gen_out)

def train_gan(key):
    print(f"Training GAN ({TARGET_TYPE})...")
    k_g, k_d, k_loop = jax.random.split(key, 3)
    
    # --------------------------------------------------------------------------
    # 1. MODEL INITIALIZATION (DYNAMIC ARCHITECTURE)
    # --------------------------------------------------------------------------
    H_GAN = get_balanced_hidden_dim(PARAMETER_BUDGET, DATA_DIM, "GAN")
    
    # Validation of Parameter Count
    n_gen = (16+1)*H_GAN + (H_GAN+1)*DATA_DIM
    n_disc = (DATA_DIM+1)*H_GAN + (H_GAN+1)*1
    total_params = n_gen + n_disc
    print(f"   -> GAN Est. Params: {total_params} (Hidden={H_GAN})")
    
    # Generator: 16 (Latent) -> Hidden -> DATA_DIM (Output)
    gen_p = init_mlp(k_g, [16, H_GAN, DATA_DIM]) 
    # Discriminator: DATA_DIM -> Hidden -> 1 (Probability Real)
    disc_p = init_mlp(k_d, [DATA_DIM, H_GAN, 1])
    
    # Optimizers: Adam with standard betas for GANs (Beta1=0.5 often helps stability)
    opt_g = optax.adam(LR_GAN, b1=0.5)
    opt_d = optax.adam(LR_GAN, b1=0.5)
    
    state_g = opt_g.init(gen_p)
    state_d = opt_d.init(disc_p)
    
    # --------------------------------------------------------------------------
    # 2. TRAINING STEP (JIT COMPILED)
    # --------------------------------------------------------------------------
    @jit
    def step(i, gp, dp, sg, sd, k):
        kz, kr, kzg, kgumbel = jax.random.split(k, 4)
        
        # === PHASE A: TRAIN DISCRIMINATOR ===
        # Goal: Maximize probability of correctly classifying Real vs Fake.
        
        real = sample_target(kr, BATCH_SIZE)          # Real Data
        z = jax.random.normal(kz, (BATCH_SIZE, 16))   # Latent Noise
        
        def d_loss(d_params, g_params, x_real, z_in):
            # 1. Generate Fake Data 
            # ------------------------------------------------------------------
            # DISCRETE GAN LOGIC:
            # If NUM_BINS > 0, the Generator output is logits for the bins.
            # We must sample appropriately to fool the Discriminator.
            # ------------------------------------------------------------------
            gen_out = forward_mlp(g_params, z_in)
            
            if NUM_BINS > 0:
                # Gumbel-Softmax: Differentiable discrete sampling
                # We use "hard=True" (Straight-Through Estimator) so the Discriminator
                # sees One-Hot vectors, matching the Real Data format.
                # Gradients still look like they came from the Softmax (ST).
                fake = gumbel_softmax(gen_out, kgumbel, temperature=1.0, hard=True)
            else:
                fake = gen_out
            
            # 2. Real Loss
            logits_real = forward_mlp(d_params, x_real)
            loss_real = optax.sigmoid_binary_cross_entropy(logits_real, jnp.ones((BATCH_SIZE, 1)) * 0.9).mean()
            
            # 3. Fake Loss
            logits_fake = forward_mlp(d_params, fake)
            loss_fake = optax.sigmoid_binary_cross_entropy(logits_fake, jnp.zeros((BATCH_SIZE, 1))).mean()
            return loss_real + loss_fake
            
        # Compute Gradients for Discriminator
        ld, gd = value_and_grad(d_loss)(dp, gp, real, z)
        ud, nsd = opt_d.update(gd, sd, dp)
        ndp = optax.apply_updates(dp, ud)
        
        # === PHASE B: TRAIN GENERATOR ===
        # Goal: Maximize D's confusion.
        
        z2 = jax.random.normal(kzg, (BATCH_SIZE, 16)) 
        
        def g_loss(g_params, d_params, z_in):
            # Same sampling logic for Generator Step
            gen_out = forward_mlp(g_params, z_in)
            if NUM_BINS > 0:
                fake = gumbel_softmax(gen_out, kgumbel, temperature=1.0, hard=True)
            else:
                fake = gen_out
            
            logits_fake = forward_mlp(d_params, fake)
            # Minimize -Log(D(fake))
            return optax.sigmoid_binary_cross_entropy(logits_fake, jnp.ones((BATCH_SIZE, 1))).mean()
            
        # Compute Gradients
        lg, gg = value_and_grad(g_loss)(gp, ndp, z2)
        ug, nsg = opt_g.update(gg, sg, gp)
        ngp = optax.apply_updates(gp, ug)
        
        return ngp, ndp, nsg, nsd, lg, ld

    # --------------------------------------------------------------------------
    # 3. TRAINING LOOP
    # --------------------------------------------------------------------------
    kld_history = []
    snapshots = {}
    checkpoints = np.linspace(0, N_EPOCHS, 11, dtype=int)[1:] 

    # Pre-sampled target for consistent KLD computation (eval set)
    target_raw = sample_target(jax.random.PRNGKey(999), 5000)
    target_kld = decode_to_1d(np.array(target_raw))
    
    for i in tqdm(range(N_EPOCHS), desc="GAN"):
        k_loop, subk, k_eval = jax.random.split(k_loop, 3)
        gen_p, disc_p, state_g, state_d, _, _ = step(i, gen_p, disc_p, state_g, state_d, subk)
        
        # Evaluation
        if i % KLD_CHECK_EVERY == 0 or i == N_EPOCHS - 1:
            samples_model = sample_gan(gen_p, k_eval, 5000)
            samples_decoded = decode_to_1d(samples_model)
            
            # 1. KLD Metric (Histogram based)
            kld = compute_kld_histogram(target_kld, samples_decoded)
            
            # 2. Wasserstein Metric (Quantile based) - More robust
            wd = compute_wasserstein_1d(target_kld, samples_decoded)
            
            kld_history.append((i, kld, wd)) # Tuple now (Epoch, KLD, WD)
        
        # Snapshotting
        if i in checkpoints or i == N_EPOCHS-1:
             snapshots[i] = gen_p
            
    return gen_p, "GAN", kld_history, snapshots

# ==============================================================================
#                               METHOD 2: VARIATIONAL AUTOENCODER (VAE)
# ==============================================================================
"""
VAE THEORETICAL FRAMEWORK
=========================

1. Concept
----------
A VAE models the data probability P(x) by introducing latent variables z.
Since the marginal P(x) = Integral P(x|z)P(z) dz is intractable, we use variational
inference to approximate the posterior P(z|x) with a network Q(z|x).

2. Evidence Lower Bound (ELBO)
------------------------------
We maximize a lower bound on the log-likelihood:
    log P(x) >= E_{z~Q} [log P(x|z)] - KL( Q(z|x) || P(z) )

    Term 1: **Reconstruction Log-Likelihood**.
            "How well does the decoder P(x|z) recreate the input?"
            For Gaussian data, this is equivalent to Mean Squared Error.
            
    Term 2: **Regularization**.
            "How close is the learned latent structure to the Prior P(z)=N(0,I)?"
            This enforces structure in the latent space, enabling valid sampling.

3. The Reparameterization Trick
-------------------------------
To backpropagate through the random sampling z ~ Q(z|x), we rewrite z as:
    z = mu + sigma * epsilon,   where epsilon ~ N(0, I)
randomness is now in 'epsilon' (external input), so gradients can flow to 'mu' and 'sigma'.
"""

def sample_vae(params, key, n_samples):
    """
    Inference Step for VAE:
    Draws latent noise z ~ P(z) = N(0,I) and passes it through Decoder.
    """
    dec_p = params[1]
    z = jax.random.normal(key, (n_samples, 2)) # Prior is always N(0, 1)
    return np.array(forward_mlp(dec_p, z))

def train_vae(key):
    print(f"Training VAE ({TARGET_TYPE})...")
    k_e, k_d, k_loop = jax.random.split(key, 3)
    
    # Architecture Config
    H_VAE = get_balanced_hidden_dim(PARAMETER_BUDGET, DATA_DIM, "VAE")
    
    # Validation
    Z_val = 2
    # Enc: (D+1)*H + (H+1)*2Z
    n_enc = (DATA_DIM+1)*H_VAE + (H_VAE+1)*(2*Z_val)
    # Dec: (Z+1)*H + (H+1)*D
    n_dec = (Z_val+1)*H_VAE + (H_VAE+1)*DATA_DIM
    print(f"   -> VAE Est. Params: {n_enc + n_dec} (Hidden={H_VAE})")

    # Encoder: Input -> Hidden -> Latent Params (Mean, LogVar)
    enc_p = init_mlp(k_e, [DATA_DIM, H_VAE, 4]) 
    dec_p = init_mlp(k_d, [2, H_VAE, DATA_DIM])
    
    opt = optax.adam(LR_VAE)
    opt_state = opt.init((enc_p, dec_p))
    
    # --------------------------------------------------------------------------
    # LOSS FUNCTION (ELBO)
    # --------------------------------------------------------------------------
    @jit
    def elbo_loss(params, x, k):
        enc, dec = params
        
        # --- A. Encoder Pass (Inference) ---
        # Predict parameters of distribution Q(z|x) = N(mu, sigma)
        stats = forward_mlp(enc, x)
        mu, logvar = stats[:, :2], stats[:, 2:]
        std = jnp.exp(0.5 * logvar)
        
        # --- B. Reparameterization Trick ---
        # z = mu + sigma * epsilon
        eps = jax.random.normal(k, std.shape)
        z = mu + std * eps
        
        # --- C. Decoder Pass (Generation) ---
        # Reconstruct x from z
        recon = forward_mlp(dec, z)
        
        # --- D. Loss Components ---
        
        # 1. Reconstruction Loss (MSE)
        # Corresponds to log P(x|z) assuming fixed variance Gaussian likelihood.
        recon_loss = jnp.mean(jnp.sum((recon - x)**2, axis=1))
        
        # 2. KL Divergence Regularizer (Analytical)
        # Closed form KL( N(mu, sigma) || N(0, 1) )
        # = -0.5 * Sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld_term = -0.5 * jnp.mean(jnp.sum(1 + logvar - mu**2 - jnp.exp(logvar), axis=1))
        
        # Weighting
        # Beta-VAE formulation: recon + beta * KL
        # beta=0.1 reduces regularization, allowing sharper reconstructions at the cost of
        # potentially less smooth latent space (trade-off).
        return recon_loss + 0.1 * kld_term 

    @jit
    def step(params, opt_st, k):
        x = sample_target(k, BATCH_SIZE)
        k_loss, k_next = jax.random.split(k)
        
        # Compute Gradients of Negative ELBO (Loss)
        l, g = value_and_grad(elbo_loss)(params, x, k_loss)
        updates, ns = opt.update(g, opt_st, params)
        np = optax.apply_updates(params, updates)
        return np, ns

    # --------------------------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------------------------
    params = (enc_p, dec_p)
    kld_history = []
    snapshots = {}
    checkpoints = np.linspace(0, N_EPOCHS, 11, dtype=int)[1:]
    
    target_raw = sample_target(jax.random.PRNGKey(999), 5000)
    target_kld = decode_to_1d(np.array(target_raw))

    for i in tqdm(range(N_EPOCHS), desc="VAE"):
        k_loop, subk, k_eval = jax.random.split(k_loop, 3)
        params, opt_state = step(params, opt_state, subk)
        
        if i % KLD_CHECK_EVERY == 0 or i == N_EPOCHS - 1:
            samples_model = sample_vae(params, k_eval, 5000)
            samples_decoded = decode_to_1d(samples_model)
            kld = compute_kld_histogram(target_kld, samples_decoded)
            wd = compute_wasserstein_1d(target_kld, samples_decoded)
            kld_history.append((i, kld, wd))
            
        if i in checkpoints or i == N_EPOCHS-1:
             snapshots[i] = params
        
    return params, "VAE", kld_history, snapshots

# ... (RBM SECTION SKIPPED FOR BREVITY - HANDLED BELOW IN EXECUTION) ...
# Actually merging RBM update here to save turns if possible or using next tool call.
# I will handle RBM in the next tool call to be safe with line numbers.

# ==============================================================================
# METHOD 3: RESTRICTED BOLTZMANN MACHINE (RBM)
# ==============================================================================
"""
RBM Theory:
-----------
An Energy-Based Model (EBM) defined by a bipartite graph (Visible units v, Hidden units h).
Likelihood:
    P(v, h) = exp(-E(v, h)) / Z
    E(v, h) = -v^T W h - b_v^T v - b_h^T h

    Where:
    - v: Visible units (Data)
    - h: Hidden units (Latent Features)
    - Z: Partition Function (Intractable sum over all states)

Optimization (Maximum Likelihood):
    We want to maximize Log P(v). The gradient is:
    grad(Log P(v)) = < grad(-E) >_{P(h|v)data}  -  < grad(-E) >_{P(v,h)model}
    
    Term 1 (Positive Phase): Clamps v to Data. Easy to compute.
    Term 2 (Negative Phase): Samples form the Model. Hard to compute (requires Z).

Contrastive Divergence (CD-k) Approximation:
    Instead of sampling from the perfect equilibrium distribution P(model), we 
    approximate the negative phase by running a Gibbs Sampling chain for 'k' steps,
    initialized at the data point itself.
    
    Gradient ~  EnergyGradient(v_data) - EnergyGradient(v_k_step_reconstruction)

Autodiff Implementation:
    We define the Free Energy F(v) = -log sum_h exp(-E(v,h)).
    Minimizing F(v_data) pushes down energy of data.
    Maximizing F(v_sample) pushes up energy of non-data.
    Loss = Mean(F(v_data)) - Mean(F(v_sample))

Theory Note (Gaussian RBM):
    A standard Gaussian-Bernoulli RBM Energy function includes a quadratic term $\sum v_i^2 / (2\sigma^2)$.
    Because we treat the variance $\sigma^2$ as fixed (hyperparameter) and not learnable, this term 
    is constant with respect to parameters W and b. Therefore, its gradient is zero and we can 
    omit it from the training objective. However, for comparing raw Energy/Free Energy values 
    across different model types, this offset would matter.
"""
def sample_rbm(params, key, n_samples):
    # Block Gibbs Sampling from noise to generate new samples
    W, b_v, b_h = params["W"], params["b_v"], params["b_h"]
    # Start random v (Noise initialization)
    v = jax.random.normal(key, (n_samples, DATA_DIM))
    
    # 1. Gibbs Sampling Setup
    def gibbs(v_curr, k):
        k1, k2 = jax.random.split(k)
        
        # A. Sample Hidden from Visible (v -> h)
        #    H units are Bernoulli (0 or 1) given V
        h_p = jax.nn.sigmoid(v_curr @ W + b_h)
        h = jax.random.bernoulli(k1, h_p).astype(jnp.float32)
        
        # B. Sample Visible from Hidden (h -> v)
        #    V units are Gaussian (Real values) given H
        v_mean = h @ W.T + b_v
        
        #    CRITICAL: Noise Scaling match.
        noise_scale = 0.5
        if TARGET_TYPE == "1D_MIX_GAUSS": noise_scale = 0.4
        elif TARGET_TYPE == "1D_BETA": noise_scale = 0.1 # Lower noise for Beta structure
        
        v_new = v_mean + jax.random.normal(k2, v_mean.shape) * noise_scale
        return v_new
    
    # 2. Run chain for 50 steps to 'burn in' (reach equilibrium)
    keys = jax.random.split(key, 50) 
    for k in keys:
        v = gibbs(v, k)
    return np.array(v)

# ==============================================================================
#                               ARCHITECTURE HELPER
# ==============================================================================
def get_balanced_hidden_dim(budget, visible_dim, model_type):
    """
    Calculates the required Hidden Dimension (H) to meet a target Parameter Budget (P).
    
    Formulas Derived:
    -----------------
    1. RBM (Visible V, Hidden H):
       Params = W(V,H) + b_v(V) + b_h(H) = V*H + V + H
       P = H(V + 1) + V
       => H = floor( (P - V) / (V + 1) )
       
    2. GAN (Data D, Noise Z=16, Hidden H):
       Generator: (Z+1)*H + (H+1)*D
       Discriminator: (D+1)*H + (H+1)*1
       Total = H(Z + 2D + 3) + D + 1
       => H = floor( (P - D - 1) / (Z + 2D + 3) )
       
    3. VAE (Data D, Latent Z=2, Hidden H):
       Encoder: (D+1)*H + (H+1)*2Z  [Mean + LogVar]
       Decoder: (Z+1)*H + (H+1)*D
       Total = H(2D + 3Z + 2) + 2Z + D
       => H = floor( (P - 2Z - D) / (2D + 3Z + 2) )
    """
    if model_type == "RBM":
        # P = H(V+1) + V
        h = (budget - visible_dim) / (visible_dim + 1)
        return int(h)
        
    elif model_type == "GAN":
        # Z is fixed to 16 in our script
        Z_gan = 16 
        # P = H(Z + 2D + 3) + D + 1
        numerator = budget - visible_dim - 1
        denominator = Z_gan + (2 * visible_dim) + 3
        h = numerator / denominator
        return int(h)
        
    elif model_type == "VAE":
        # Z is fixed to 2 in our script
        Z_vae = 2
        # P = H(2D + 3Z + 2) + 2Z + D
        numerator = budget - (2 * Z_vae) - visible_dim
        denominator = (2 * visible_dim) + (3 * Z_vae) + 2
        h = numerator / denominator
        return int(h)
        
    return 16 # Default Fallback  
def train_rbm(key):
    print(f"Training RBM ({TARGET_TYPE})...")
    
    # Target: ~310 Params
    H_RBM = get_balanced_hidden_dim(PARAMETER_BUDGET, DATA_DIM, "RBM")
    
    # Validation
    # P = D*H + D + H
    n_rbm = DATA_DIM*H_RBM + DATA_DIM + H_RBM
    print(f"   -> RBM Est. Params: {n_rbm} (Hidden={H_RBM})")
    
    k_w, k_vb, k_hb, k_loop = jax.random.split(key, 4)
    
    # Initialization (Small random weights)
    W = jax.random.normal(k_w, (DATA_DIM, H_RBM)) * 0.01
    b_v = jnp.zeros(DATA_DIM)
    b_h = jnp.zeros(H_RBM)
    
    params = {"W": W, "b_v": b_v, "b_h": b_h}
    opt = optax.sgd(LR_RBM) # SGD is standard (and stable) for RBMs
    opt_st = opt.init(params)
    
    # Free Energy Function: F(v) = -log sum_h exp(-E(v,h))
    # Analytically integrating out the binary hidden units.
    def free_energy(p, v):
        wx_b = v @ p["W"] + p["b_h"]
        # softplus(x) = log(1 + exp(x)) matches the sum over binary states
        hidden_term = jnp.sum(jax.nn.softplus(wx_b), axis=1) 
        v_term = v @ p["b_v"]
        return -hidden_term - v_term

    @jit
    def cd_step(p, opt_st, k):
        # Contrastive Divergence - 1 Step (CD-1)
        # Start chain at data points (x_real)
        x_real = sample_target(k, BATCH_SIZE)
        
        # 1. Positive Phase (Data Driven)
        #    Infer H from V_data
        h_prob = jax.nn.sigmoid(x_real @ p["W"] + p["b_h"])
        k1, k2 = jax.random.split(k)
        h_sample = jax.random.bernoulli(k1, h_prob).astype(jnp.float32)
        
        # 2. Negative Phase (Reconstruction / Model Driven)
        #    Reconstruct V from H (Gibbs Step 1)
        v_mean = h_sample @ p["W"].T + p["b_v"]
        
        #    Noise Scaling Match
        noise_scale = 0.5
        if TARGET_TYPE == "1D_MIX_GAUSS": noise_scale = 0.4
        elif TARGET_TYPE == "1D_BETA": noise_scale = 0.1
        
        v_sample = v_mean + jax.random.normal(k2, v_mean.shape) * noise_scale
        
        #    CRITICAL IMPLEMENTATION DETAIL: Stop Gradient
        #    In CD, we treat the negative samples 'v_sample' as fixed targets 
        #    derived from the current model state. We do NOT want to backpropagate 
        #    through the sampling process itself to optimize 'where the sample lands'.
        #    We only want to optimize the Energy surface relative to these points.
        v_sample = jax.lax.stop_gradient(v_sample)
        
        # 3. CD Loss via Autodiff
        #    Minimize F(data) - F(sample).
        #    This lowers the energy of the data and raises the energy of the samples.
        def loss_fn(curr_p):
            return jnp.mean(free_energy(curr_p, x_real)) - jnp.mean(free_energy(curr_p, v_sample)) 
            
        g = grad(loss_fn)(p)
        
        # Update
        updates, ns = opt.update(g, opt_st, p)
        np = optax.apply_updates(p, updates)
        return np, ns

    kld_history = []
    snapshots = {}
    checkpoints = np.linspace(0, N_EPOCHS, 11, dtype=int)[1:]
    
    target_raw = sample_target(jax.random.PRNGKey(999), 5000)
    target_kld = decode_to_1d(np.array(target_raw))

    for i in tqdm(range(N_EPOCHS), desc="RBM"):
        k_loop, subk, k_eval = jax.random.split(k_loop, 3)
        params, opt_st = cd_step(params, opt_st, subk)
        
        if i % KLD_CHECK_EVERY == 0 or i == N_EPOCHS - 1:
            samples_model = sample_rbm(params, k_eval, 5000)
            samples_decoded = decode_to_1d(samples_model)
            kld = compute_kld_histogram(target_kld, samples_decoded)
            wd = compute_wasserstein_1d(target_kld, samples_decoded)
            kld_history.append((i, kld, wd))
            
        # Save Snapshot
        if i in checkpoints or i == N_EPOCHS-1:
             snapshots[i] = params
        
    return params, "RBM", kld_history, snapshots

# ==============================================================================
# PLOTTING & MAIN
# ==============================================================================
def sample_model(model_data, key, n_samples=5000):
    params, mode, *rest = model_data 
    if mode == "GAN": return sample_gan(params, key, n_samples)
    elif mode == "VAE": return sample_vae(params, key, n_samples)
    elif mode == "RBM": return sample_rbm(params, key, n_samples)
    return np.zeros((n_samples, 2))

# ==============================================================================
#                               PLOTTING & VISUALIZATION
# ==============================================================================
def plot_kde(results, filename="classical_benchmark_comparison_kde.png", title_suffix=""):
    """
    Generates Comparison Plot using Smooth Kernel Density Estimations (KDE).
    
    Visualization:
    --------------
    - **2D Data**: Uses filled contours (contourf).
    - **1D Data**: Uses line plots for PDF and filled area for Model density.
    
    Components:
    -----------
    1. **Target**: The Ground Truth distribution (Green lines/contours).
    2. **Models**: The Learned distribution (colored heatmaps/areas).
    3. **KLD Score**: Displayed in the title for quick comparison.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    titles = ["Target", "GAN", "VAE", "RBM"]
    
    # --- Pre-calculate Theoretical PDF Grid ---
    # --- Pre-calculate Theoretical PDF Grid ---
    if DATA_DIM == 2:
        # Grid for Contours
        delta = 0.05
        grid = jnp.arange(-3.5, 3.5, delta)
        X, Y = jnp.meshgrid(grid, grid)
        Z = jnp.zeros_like(X)
        
        if TARGET_TYPE == "2D_GMM":
             # explicit GMM sum
             # Need to use scipy.stats.multivariate_normal since JAX jax.scipy.stats.multivariate_normal.pdf is simpler?
             # Or just manual: exp(-0.5 * (x-mu)^T Sinv (x-mu)) / sqrt((2pi)^k det(S))
             # Let's use scipy for plotting simplicity/robustness
             from scipy.stats import multivariate_normal
             pos = np.dstack((X, Y))
             for weight, mean, cov in GMM_COMPONENTS:
                 # Note: GMM_COMPONENTS used jax arrays, cast to numpy for scipy
                 rv = multivariate_normal(mean=np.array(mean), cov=np.array(cov))
                 Z += weight * rv.pdf(pos)
        else:
            # Assumes 2D_MIX_GAUSS standard isotropic
            for m in MEANS:
                 Z += jnp.exp(-((X - m[0])**2 + (Y - m[1])**2) / (2 * SIGMA**2))
    else:
        # 1D Grid
        grid = jnp.linspace(-3.5, 3.5, 500)
        Z = jnp.zeros_like(grid)
        if "1D" in TARGET_TYPE and "BETA" not in TARGET_TYPE and "POISSON" not in TARGET_TYPE:
            for m in MEANS:
                 Z += jnp.exp(-(grid - m[0])**2 / (2 * SIGMA**2))
            Z = Z / Z.sum() / (grid[1]-grid[0])
        elif TARGET_TYPE == "1D_BETA":
            from scipy.stats import beta
            x_equiv = (grid / 5.0) + 0.5
            mask = (x_equiv >= 0) & (x_equiv <= 1)
            vals = beta.pdf(x_equiv[mask], BETA_A, BETA_B)
            Z = Z.at[mask].set(vals / 5.0)
            
        elif TARGET_TYPE == "1D_POISSON":
            from scipy.stats import poisson
            # grid values are 'shifted'. Original k = grid + LAMBDA
            # We check which grid points are close to integers
            # But for simple visualization, let's just evaluate PDF at grid points that map to k
            k_vals = grid + POISSON_LAMBDA
            # Poisson is defined for k >= 0 integers
            # We can plot it as a step function or points
            # For smooth 'Target' line style, we might just interpolate
            vals = poisson.pmf(np.round(k_vals), POISSON_LAMBDA)
            Z = jnp.array(vals) 
            # Zero out invalid regions (negative k)
            Z = jnp.where(k_vals < 0, 0, Z)

    # --- Plotting Loop ---
    for i, ax in enumerate(axes):
        if DATA_DIM == 2:
            ax.set_aspect('equal')
            ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
            
            if i == 0:
                ax.set_title(titles[i])
                ax.contourf(X, Y, Z, levels=20, cmap='viridis')
            else:
                res = results[i-1] 
                _, _, history, _ = res
                if len(history) > 0:
                    final_kld = history[-1][1]
                    final_wd = history[-1][2]
                else: 
                     final_kld, final_wd = 0.0, 0.0
                ax.set_title(f"{titles[i]}\nKLD: {final_kld:.4f} | WD: {final_wd:.4f}{title_suffix}")
                key = jax.random.PRNGKey(i*999)
                samples = sample_model(res, key)
                try:
                    kde = gaussian_kde(samples.T)
                    Z_est = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
                    ax.contourf(X, Y, Z_est, levels=20, cmap='magma')
                except:
                    ax.hist2d(samples[:,0], samples[:,1], bins=64, range=[[-3.5, 3.5], [-3.5, 3.5]], cmap='magma', density=True)
            # Reference
            ax.contour(X, Y, Z, levels=3, colors='green', alpha=0.3, linewidths=1)
            
        else: # 1D Plot
            ax.set_xlim(-3.5, 3.5)
            ax.set_ylim(bottom=0)
            
            # Plot Theoretical PDF
            ax.plot(grid, Z, color='green', linewidth=2, linestyle='--', label='Target')
            
            if i == 0:
                ax.set_title(titles[i])
            else:
                res = results[i-1] 
                _, _, history, _ = res
                if len(history) > 0:
                    final_kld = history[-1][1]
                    final_wd = history[-1][2]
                else:
                    final_kld, final_wd = 0.0, 0.0
                ax.set_title(f"{titles[i]}\nKLD: {final_kld:.4f} | WD: {final_wd:.4f}{title_suffix}")
                key = jax.random.PRNGKey(i*999)
                samples = sample_model(res, key)
                
                # KDE for Model
                try:
                    kde = gaussian_kde(samples.T)
                    Z_est = kde(grid)
                    ax.fill_between(grid, Z_est, alpha=0.5, color='orange', label='Model')
                except:
                    ax.hist(samples[:,0], bins=50, density=True, alpha=0.5, color='orange')
            ax.legend()

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{filename}")
    plt.show()
    plt.close()
    print(f"Saved {OUT_DIR}/{filename}")

    # Decode One-Hot if needed
    # (Target curve Z is already computed theoretically above)
    
    for i, res in enumerate(results):
        params, name, _, _ = res
        ax = axes[i]
        
        # Sample Model
        key = jax.random.PRNGKey(i*999)
        if name == "GAN": samples = sample_gan(params, key, 10000)
        elif name == "VAE": samples = sample_vae(params, key, 10000)
        elif name == "RBM": samples = sample_rbm(params, key, 10000)
        
        # Decode if Binned
        samples = decode_to_1d(samples)

        # Plot Logic (Same as before)
        if samples.shape[1] == 2:
           # 2D Hist
           ax.hist2d(samples[:,0], samples[:,1], bins=50, range=[[-3,3],[-3,3]], density=True, cmap="Blues")
           ax.set_title(f"{name} (2D Hist)")
        else:
           # 1D Hist
           ax.set_xlim(-3.5, 3.5)
           if i == 0: # Target is passed as 'results' here? No, 'results' is list of models. Target is plotted in loop?
              # Logic correction: plot_hist iterates "results". The original code had specific logic for target on i=0?
              # Let's check the original code context.
              pass
           
           # Actually, standardizing the loop:
           # We plot Target on every ax as reference? Or distinct axes?
           # Previous code: 
           # if i==0: plot Target; else: plot results[i-1] ?
           # Let's Stick to the logic in file.
           pass 
           
# STOP. The previous 'plot_hist' implementation had a 4-panel setup where i=0 was Target.
# I need to match that structure.

# ==============================================================================
#                               HELPER: VAE MANIFOLD
# ==============================================================================
def plot_vae_manifold(decoder_params, filename="vae_latent_manifold_sweep.png"):
    """
    Visualizes the Generative Manifold (Latent Space Structure) of the VAE.
    
    THEORY:
    -------
    A key success criterion for VAEs is learning a CONTINUOUS latent representation, 
    unlike vanilla Autoencoders (AE) which often learn disjoint 'islands' (Memorization).
    
    This function performs a "Latent Sweep":
    1. Grid Sampling: We iterate z1, z2 from -3 to +3 (covering the N(0,1) prior mass).
    2. Decoding: We generate the expected output E[x|z] for each point.
    
    INTERPRETATION:
    ---------------
    - Smooth Gradient in plot = SUCCESS. The model interpolates semantically between modes.
    - Sharp/Noisy Transitions = FAILURE. The model has 'holes' or has failed to regularize.
    """
    print("Generating VAE Latent Manifold Sweep...")
    
    # 1. Create Grid
    n = 20
    grid_x = np.linspace(-3, 3, n)
    grid_y = np.linspace(-3, 3, n)
    
    # If 1D Output (most of our cases), we plot a Heatmap of the OUTPUT VALUE
    # If 2D Output, we might plot a grid of 2D points? (Harder to viz)
    # Let's assume 1D Output for the Heatmap visualization.
    
    # Support 2D Output visualization
    n_dims = DATA_DIM if NUM_BINS == 0 else 1 # If binned, we decode to 1D value
    
    if n_dims == 1:
        # 1D Case: Plot Heatmap of the single output variable
        # ----------------------------------------------------------------------
        # THEORY: 1D MANIFOLD
        # ----------------------------------------------------------------------
        # For a 1D target (e.g. Trimodal), we map (z1, z2) -> x (scalar).
        # We visualize this as a Heatmap where Color = Output Value x.
        #
        # - Vertical Stripes: z1 controls x (z2 is ignored).
        # - Horizontal Stripes: z2 controls x (z1 is ignored).
        # - Smooth Gradient: The latent space creates a continuous path between modes.
        # ----------------------------------------------------------------------
        canvas = np.zeros((n, n))
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = jnp.array([[xi, yi]])
                x_out = forward_mlp(decoder_params, z_sample)
                
                # If Discrete: Take Expectation (Softmax * Values) or Argmax
                if NUM_BINS > 0:
                    probs = jax.nn.softmax(x_out)
                    centers = jnp.linspace(-3.5, 3.5, NUM_BINS)
                    val = jnp.sum(probs * centers)
                else:
                    val = x_out[0, 0]
                canvas[i, j] = val

        plt.figure(figsize=(8, 6))
        plt.imshow(canvas, origin="lower", extent=[-3, 3, -3, 3], cmap="viridis")
        plt.colorbar(label="Generated Output x")
        plt.title(f"VAE Latent Manifold Sweep (z1 vs z2) -> x\nTarget: {TARGET_TYPE}")
        plt.xlabel("Latent z1")
        plt.ylabel("Latent z2")
        plt.savefig(f"{OUT_DIR}/{filename}")
        plt.close()
        
    elif n_dims == 2:
        # 2D Case: Plot Dimension 0 (X) and Dimension 1 (Y) separately
        # ----------------------------------------------------------------------
        # THEORY: VISUALIZING DISENTANGLEMENT
        # ----------------------------------------------------------------------
        # In a 2D multimodal problem (like 4 clusters at corners), the VAE often 
        # learns a "Disentangled Representation":
        # - One latent variable (e.g. z1) controls the X-coordinate.
        # - The other latent variable (e.g. z2) controls the Y-coordinate.
        #
        # If we only plot the first output dimension (as in a standard 1D sweep),
        # we would only see dependency on z1, making z2 appear "useless" or collapsed.
        # BY PLOTTING BOTH DIMENSIONS SEPARATELY, we reveal the full structure:
        # - Panel 1 (Output X) should show a gradient along z1 (vertical stripes).
        # - Panel 2 (Output Y) should show a gradient along z2 (horizontal stripes).
        # ----------------------------------------------------------------------
        canvas_x = np.zeros((n, n))
        canvas_y = np.zeros((n, n))
        
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = jnp.array([[xi, yi]])
                x_out = forward_mlp(decoder_params, z_sample) # Shape (1, 2)
                canvas_x[i, j] = x_out[0, 0]
                canvas_y[i, j] = x_out[0, 1]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        im1 = axes[0].imshow(canvas_x, origin="lower", extent=[-3, 3, -3, 3], cmap="viridis")
        axes[0].set_title("Generated Output X (Dim 0)")
        axes[0].set_xlabel("Latent z1"); axes[0].set_ylabel("Latent z2")
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        
        im2 = axes[1].imshow(canvas_y, origin="lower", extent=[-3, 3, -3, 3], cmap="viridis")
        axes[1].set_title("Generated Output Y (Dim 1)")
        axes[1].set_xlabel("Latent z1"); axes[1].set_ylabel("Latent z2")
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        
        plt.suptitle(f"VAE Latent Manifold Sweep (z1 vs z2) -> x,y\nTarget: {TARGET_TYPE}")
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/{filename}")
        plt.close()
    print(f"Saved {OUT_DIR}/{filename}")
    
def plot_hist(results, filename="classical_benchmark_comparison_hist.png"):
    """
    Generates Comparison Plot using Raw Sample Histograms.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    titles = ["Target", "GAN", "VAE", "RBM"]
    
    for i, ax in enumerate(axes):
        # Decode Data for Plotting
        if i == 0:
            # Target
            data = np.array(sample_target(jax.random.PRNGKey(0), 10000))
            data = decode_to_1d(data)
            color = 'purple'
            label = 'Target'
        else:
            # Models
            res = results[i-1]
            _, name, history, _ = res
            if len(history) > 0:
                kld = history[-1][1]
                wd = history[-1][2]
            else:
                kld, wd = 0.0, 0.0
            
            ax.set_title(f"{titles[i]}\nKLD: {kld:.4f} | WD: {wd:.4f}")
            
            key = jax.random.PRNGKey(i*999)
            data = sample_model(res, key) # Helper needed or direct?
            # Direct calls:
            params, m_name, _, _ = res
            if m_name == "GAN": data = sample_gan(params, key, 10000)
            elif m_name == "VAE": data = sample_vae(params, key, 10000)
            elif m_name == "RBM": data = sample_rbm(params, key, 10000)
            
            data = decode_to_1d(data)
            color = 'blue'
            label = m_name

        # Plotting
        if data.shape[1] == 2:
            ax.set_aspect('equal')
            ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
            ax.hist2d(data[:,0], data[:,1], bins=64, range=[[-3.5, 3.5], [-3.5, 3.5]], cmap='magma', density=True)
            if i == 0: ax.set_title("Target")
        else:
             ax.set_xlim(-3.5, 3.5)
             ax.hist(data[:,0], bins=64, density=True, color=color, alpha=0.7, label=label)
             if i == 0: ax.set_title("Target")
             ax.legend()
             
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{filename}")
    plt.show()
    plt.close()
    print(f"Saved {OUT_DIR}/{filename}")

def plot_kld_history(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for res in results:
         _, name, history, _ = res
         steps = [h[0] for h in history]
         klds = [h[1] for h in history]
         wds = [h[2] for h in history] # Wasserstein
         
         ax1.plot(steps, klds, label=f"{name} (Min: {min(klds):.3f})", linewidth=2)
         ax2.plot(steps, wds, label=f"{name} (Min: {min(wds):.3f})", linewidth=2)
         
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("KL Divergence (Log)")
    ax1.set_title("Metric 1: KL Divergence (Lower is Better)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")
    
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Wasserstein Dist (Log)")
    ax2.set_title("Metric 2: Wasserstein Distance (Lower is Better)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")
    
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/metrics_comparison.png")
    plt.show()
    print(f"Saved {OUT_DIR}/metrics_comparison.png")

def main():
    key = jax.random.PRNGKey(SEED)
    k1, k2, k3 = jax.random.split(key, 3)
    
    print(f"\n=== STARTING CLASSICAL BENCHMARKS ({TARGET_TYPE}) ===")
    print(f"Data Dimension: {DATA_DIM}")
    print(f"Epochs: {N_EPOCHS}")
    print(f"Epochs: {N_EPOCHS}")
    if "POISSON" in TARGET_TYPE:
        print(f"Poisson Lambda: {POISSON_LAMBDA}")
    elif "BETA" in TARGET_TYPE:
        print(f"Beta Params: {BETA_A}, {BETA_B}")
    else:
        print(f"Target Sigma: {SIGMA}")
    print(f"Parameter Budget: ~310 (Hidden Dims adjusted)")
    
    gan_res = train_gan(k1)
    vae_res = train_vae(k2)
    rbm_res = train_rbm(k3)
    

    
    results = [gan_res, vae_res, rbm_res]
    
    # 1. Final Plots
    # plot_kde(results, filename="classical_benchmark_comparison_kde.png") # Removed KDE as Hist is better for discrete
    plot_hist(results, filename="classical_benchmark_comparison_hist.png")
    plot_kld_history(results)
    
    # 2. VAE Manifold (Special Visualization)
    # vae_res[0] is (enc_params, dec_params)
    vae_decoder = vae_res[0][1]
    plot_vae_manifold(vae_decoder)
    
    # 2. Timeline Plots (SNAPSHOT_PERCENT Intervals)
    print(f"\nGenerating Timeline Plots ({SNAPSHOT_PERCENT}% intervals)...")
    n_snaps = 100 // SNAPSHOT_PERCENT
    checkpoints = np.linspace(0, N_EPOCHS, n_snaps+1, dtype=int)[1:]
    
    for epoch in checkpoints:
        # Construct ephemeral result objects for this epoch
        res_epoch = []
        for res in results:
             _, name, hist, snaps = res
             # Get params at this epoch
             if epoch in snaps:
                 p_at_epoch = snaps[epoch]
             else:
                 p_at_epoch = snaps[list(snaps.keys())[-1]]
             
             # Find KLD at this epoch
             kld = np.nan # Default to NaN if not found
             
             # Search for CLOSEST match (to satisfy "based on percentage")
             if len(hist) > 0:
                 closest_entry = min(hist, key=lambda x: abs(x[0] - epoch))
                 curr_kld = closest_entry[1]
                 curr_wd = closest_entry[2]
             else:
                 curr_kld, curr_wd = 0.0, 0.0
             
             # Dummy history for plotting function (i, kld, wd)
             res_epoch.append((p_at_epoch, name, [(epoch, curr_kld, curr_wd)], {}))
             
        pct = int((epoch / N_EPOCHS) * 100)
        plot_kde(res_epoch, filename=f"timeline_kde_epoch_{epoch}_{pct}pct.png", title_suffix=f"\n(Epoch {epoch}, {pct}%)")
        
    print("\nBenchmark Complete. Results saved to ./results_classical_benchmarks/")

if __name__ == "__main__":
    main()
