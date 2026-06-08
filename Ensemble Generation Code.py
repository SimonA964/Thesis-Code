import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
from scipy.stats import gamma as gamma_dist
from scipy import stats

# Parameters
num_samples = 1000
N = 10
beta = 1
m = 1
seed = None
n_steps = 100000
step_size = 7
step_size_p = 5
mu = 0
sigma = 0.1


# Generate tensor J
def generate_J(N, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Generate J from normal distribution
    J = np.random.normal(mu, sigma, size=(N, N, N))

    # Rescale
    mean_J2 = np.mean(J ** 2)
    J *= np.sqrt((1.0 / N) / mean_J2)
    return J

J = generate_J(N, seed=seed)
#Alpha rescaling
J*= 0.0025

# Computes the interaction term used in the potential energy V
@njit
def compute_S(x, J):
    N = len(x)
    S = np.zeros(N)
    for a in range(N):
        for i in range(N):
            for j in range(N):
                S[a] += J[a, i, j] * x[i] * x[j]
    return S

#Metropolis-Hastings MCMC sampler for positions and momenta.
#Alternates between position and momentum updates with equal probability.
@njit
def sample_positions_fast(x_init, J, beta, m, n_steps, step_size, n_burnin=90000):
    N = len(x_init)
    x = x_init.copy()
    #Initialise momenta from Maxwell distribution
    p = np.random.randn(N) * np.sqrt(m / beta)

    #Compute initial interaction terms and energies
    S = compute_S(x, J)
    V = np.sum(S * S)
    KE = np.sum(p * p) / (2 * m)

    #Burn-in phase
    for _ in range(n_burnin):
        if np.random.rand() < 0.5:
            k = np.random.randint(N)
            delta = step_size * np.random.randn()
            T = np.zeros(N)
            for a in range(N):
                for j in range(N):
                    T[a] += J[a, k, j] * x[j]
            S_trial = S.copy()
            for a in range(N):
                S_trial[a] += 2.0 * delta * T[a] + delta * delta * J[a, k, k]
            V_trial = np.sum(S_trial * S_trial)
            dV = V_trial - V
            if dV < 0.0 or np.random.rand() < np.exp(-beta * dV):
                x[k] += delta
                S = S_trial
                V = V_trial
        else:
            #Momentum update
            k = np.random.randint(N)
            p_trial = p.copy()
            p_trial[k] += step_size_p * np.random.randn()
            KE_trial = np.sum(p_trial * p_trial) / (2 * m)
            dKE = KE_trial - KE
            if dKE < 0.0 or np.random.rand() < np.exp(-beta * dKE):
                p = p_trial
                KE = KE_trial

    # Sampling phase
    n_accepted_x = 0
    n_accepted_p = 0
    for _ in range(n_steps):
        if np.random.rand() < 0.5:
            k = np.random.randint(N)
            delta = step_size * np.random.randn()
            T = np.zeros(N)
            for a in range(N):
                for j in range(N):
                    T[a] += J[a, k, j] * x[j]
            S_trial = S.copy()
            for a in range(N):
                S_trial[a] += 2.0 * delta * T[a] + delta * delta * J[a, k, k]
            V_trial = np.sum(S_trial * S_trial)
            dV = V_trial - V
            if dV < 0.0 or np.random.rand() < np.exp(-beta * dV):
                x[k] += delta
                S = S_trial
                V = V_trial
                n_accepted_x += 1
        else:
            k = np.random.randint(N)
            p_trial = p.copy()
            p_trial[k] += step_size_p * np.random.randn()
            KE_trial = np.sum(p_trial * p_trial) / (2 * m)
            dKE = KE_trial - KE
            if dKE < 0.0 or np.random.rand() < np.exp(-beta * dKE):
                p = p_trial
                KE = KE_trial
                n_accepted_p +=1

    #Compute acceptance rates as percentages
    acceptance_rate_x = n_accepted_x / (n_steps / 2) * 100
    acceptance_rate_p = n_accepted_p / (n_steps / 2) * 100
    return x, p, V, acceptance_rate_x, acceptance_rate_p


# Parallel ensemble generation
@njit(parallel=True)
def generate_ensemble(xs, ps, energies, acceptance_rate_x, acceptance_rate_p, J, beta, m, n_steps, step_size):
    num_samples, N = xs.shape

    for i in prange(num_samples):
        #Avoids correlations between samples
        np.random.seed(i * 13246 + 7891011)
        x0 = 0.1 * np.random.randn(N)
        x_final, p_final, V_final, acc_rate_x, acc_rate_p = sample_positions_fast(x0, J, beta, m, n_steps, step_size)
        xs[i] = x_final
        ps[i] = p_final
        energies[i] = V_final
        acceptance_rate_x[i] = acc_rate_x
        acceptance_rate_p[i] = acc_rate_p

# Generate ensemble
xs = np.zeros((num_samples, N))
ps = np.zeros((num_samples, N))
energies = np.zeros(num_samples)
acceptance_rate_x = np.zeros(num_samples)
acceptance_rate_p = np.zeros(num_samples)

generate_ensemble(xs, ps, energies,acceptance_rate_x, acceptance_rate_p, J, beta, m, n_steps, step_size)


#Save
np.savez(f"Testthesis_thermal_N{N}_samp{num_samples}_beta{beta}_accept{np.mean(acceptance_rate_x)}_accept{np.mean(acceptance_rate_p)}.npz",
         x=xs, p=ps, J=J)

#Figure 1: Acceptance rates
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

a1x = axes[0]
a1x.plot(acceptance_rate_x, '.', markersize=3)
a1x.set_xlabel("Sample index", fontsize=20)
a1x.set_ylabel("Acceptance rate (%)", fontsize=20)
a1x.tick_params(axis='both', labelsize=18)
a1x.text(0.02, 0.98, '(a)', transform=a1x.transAxes,
         fontsize=20, fontweight='normal', va='top', ha='left', zorder=5)
print(np.mean(acceptance_rate_x))

a2x = axes[1]
a2x.plot(acceptance_rate_p, '.', markersize=3)
a2x.set_xlabel("Sample index", fontsize=20)
a2x.set_ylabel("Acceptance rate (%)", fontsize=20)
a2x.tick_params(axis='both', labelsize=18)
a2x.text(0.02, 0.98, '(b)', transform=a2x.transAxes,
         fontsize=20, fontweight='normal', va='top', ha='left', zorder=5)
print(np.mean(acceptance_rate_p))

plt.tight_layout()
plt.show()
plt.savefig('acceptance_rates.pdf', format='pdf', bbox_inches='tight')\


#Figure 2: Positions and momenta for 5 samples
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle(f"Sample Positions & Momenta  N={N}, β={beta}", fontsize=14)

np.random.seed(5)
indices = np.random.choice(num_samples, 5, replace=False)

ax = axes[0]
for idx in indices:
    ax.plot(xs[idx], marker='o', linestyle='-', alpha=0.8, label=f"Sample {idx}")
ax.set_xlabel("Coordinate index i")
ax.set_ylabel("x_i")
ax.set_title("Positions for 5 samples")
ax.legend(fontsize=7)

ax = axes[1]
for idx in indices:
    ax.plot(ps[idx], marker='o', linestyle='-', alpha=0.8, label=f"Sample {idx}")
ax.set_xlabel("Coordinate index i")
ax.set_ylabel("p_i")
ax.set_title("Momenta for 5 samples")
ax.legend(fontsize=7)

fig.tight_layout(rect=[0, 0, 1, 0.95])

#Figure 3: Energy diagnostics
fig, axes = plt.subplots(1, 3, figsize=(22, 6))

ax1 = axes[0]
ax1.plot(energies, '.', markersize=3)
ax1.set_xlabel("Sample index", fontsize=24)
ax1.set_ylabel(r"Energy ($\epsilon$)", fontsize=24)
ax1.tick_params(axis='both', labelsize=22)
ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes,
         fontsize=24, fontweight='normal', va='top', ha='left', zorder=5)
print(np.mean(energies))
ax2 = axes[1]
ax2.hist(energies, bins=50, density=True, alpha=0.7, label="Sampled $V(x)$")
E_range = np.linspace(energies.min(), energies.max(), 500)
k_shape, scale = N / 2, 1.0 / beta
boltzmann_pdf = gamma_dist.pdf(E_range, a=k_shape, scale=scale)
ax2.plot(E_range, boltzmann_pdf, 'r-', linewidth=2, label="Fitted distribution")
ax2.set_xlabel(r"Energy ($\epsilon$)", fontsize=24)
ax2.set_ylabel("Probability density", fontsize=24)
ax2.tick_params(axis='both', labelsize=22)
ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes,
         fontsize=24, fontweight='normal', va='top', ha='left', zorder=5)
ax2.legend(fontsize=18)

ax3 = axes[2]
kinetic_energies = np.sum(ps**2, axis=1) / (2 * m)
ax3.hist(energies,         bins=50, density=True, alpha=0.5, label=f"Potential  $\\langle V \\rangle = {energies.mean():.3f}$")
ax3.hist(kinetic_energies, bins=50, density=True, alpha=0.5, label=f"Kinetic  $\\langle T \\rangle = {kinetic_energies.mean():.3f}$")
ax3.set_xlabel(r"Energy ($\epsilon$)", fontsize=24)
ax3.set_ylabel("Probability density", fontsize=24)
ax3.tick_params(axis='both', labelsize=22)
ax3.text(0.02, 0.98, '(c)', transform=ax3.transAxes,
         fontsize=24, fontweight='normal', va='top', ha='left', zorder=5)
ax3.legend(fontsize=18)

plt.tight_layout()
plt.savefig('energy diagnostic.pdf', format='pdf', bbox_inches='tight')
plt.close()

#Figure 4: Covariance matrices
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

xs_c = xs - xs.mean(axis=0)
ps_c = ps - ps.mean(axis=0)
C_xx = (xs_c @ xs_c.T) / (N - 1)
C_pp = (ps_c @ ps_c.T) / (N - 1)

for ax, C, label, xlabel, ylabel in zip(
    axes,
    [C_xx, C_pp],
    ['(a)', '(b)'],
    ['Sample index $s_2$', 'Sample index $s_2$'],
    ['Sample index $s_1$', 'Sample index $s_1$'],
):
    im = ax.imshow(C, cmap='RdBu', aspect='auto',
                   vmin=-np.abs(C).max(), vmax=np.abs(C).max())
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=16)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(axis='both', labelsize=18)
    ax.text(0.02, 0.98, label, transform=ax.transAxes,
            fontsize=20, fontweight='normal', va='top', ha='left', zorder=5)

plt.tight_layout()
plt.savefig('covariance_matrices.pdf', format='pdf', bbox_inches='tight')
plt.close()
#Interactive HTML exports
import plotly.express as px

fig_xx = px.imshow(C_xx, color_continuous_scale='RdBu', origin='lower',
                   title="Position covariance (sample vs sample)")
fig_xx.update_layout(xaxis_title="Sample index s2", yaxis_title="Sample index s1")
fig_xx.write_html("C_xx_interactive.html")

fig_pp = px.imshow(C_pp, color_continuous_scale='RdBu', origin='lower',
                   title="Momentum covariance (sample vs sample)")
fig_pp.update_layout(xaxis_title="Sample index s2", yaxis_title="Sample index s1")
fig_pp.write_html("C_pp_interactive.html")

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(J.flatten(), bins=50, density=True, alpha=0.7)
ax.set_xlabel(r"J coupling ($\xi$)", fontsize=20)
ax.set_ylabel("Probability density", fontsize=20)
ax.tick_params(axis='both', labelsize=18)
plt.tight_layout()
plt.savefig('J_distribution.pdf', format='pdf', bbox_inches='tight')
plt.close()

print(f"Mean potential:          {energies.mean():.4f}")
print(f"Mean kinetic:            {kinetic_energies.mean():.4f}")
print(f"Expected N/(2β):         {N / (2 * beta):.4f}")

