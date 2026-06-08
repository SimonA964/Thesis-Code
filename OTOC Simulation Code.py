# N number of particles (Leapfrog)
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from numba import njit, prange
import time as tm

# Parameters
N_subset = 10
num_samples_subset = 5000

dt = 0.00125
dt_back = -dt
steps_back = 4800
returnto_step = 4800
steps_forward = 6400
steps_g = 1
m = 1.0
pert = 0.1

# Load thermal ensemble
data = np.load("Testthesis_thermal_N10_samp1000_beta1000_accept52.231366_accept57.399373999999995.npz")
xs_full = data['x']
ps_full = data['p']
J_full = data['J']

# Subset selection
def get_subset(xs_full, ps_full, J_full, num_samples=None, N=None):
    if num_samples is None:
        num_samples = xs_full.shape[0]
    if N is None:
        N = xs_full.shape[1]

    xs_sub = xs_full[:num_samples, :N].copy()
    ps_sub = ps_full[:num_samples, :N].copy()
    J_sub = J_full[:N, :N, :N].copy()

    return xs_sub, ps_sub, J_sub

xs_sub, ps_sub, J_sub = get_subset(
    xs_full, ps_full, J_full,
    num_samples=num_samples_subset,
    N=N_subset
)

# Precompute symmetric tensor
K_sub = J_sub + J_sub.transpose(0, 2, 1)

# Optimized Force
@njit(fastmath=True)
def dH_dx(x, J, K):
    N = x.shape[0]

    # Compute S_a
    S = np.zeros(N)
    for a in range(N):
        s = 0.0
        for i in range(N):
            xi = x[i]
            for j in range(N):
                s += J[a, i, j] * xi * x[j]
        S[a] = s

    # Compute gradient
    grad = np.zeros(N)
    for k in range(N):
        g = 0.0
        for a in range(N):
            Sa = S[a]
            row_sum = 0.0
            for j in range(N):
                row_sum += K[a, k, j] * x[j]
            g += Sa * row_sum
        grad[k] = -2.0 * g

    return grad

# Leapfrog Integrator
@njit(fastmath=True)
def leapfrog_step(x, p, dt, J, K, m):
    p = p + 0.5 * dt * dH_dx(x, J, K)
    x = x + dt * (p / m)
    p = p + 0.5 * dt * dH_dx(x, J, K)
    return x, p

# Parallel Simulation
@njit(fastmath=True, parallel=True)
def simulation_loop(xs, ps, J, K, m, dt,
                    steps_back, dt_back,
                    returnto_step, steps_g,
                    pert, steps_forward):

    num_samples = xs.shape[0]
    N = xs.shape[1]

    delta_p1L = np.zeros((num_samples, steps_forward))

    for s in prange(num_samples):

        xL0 = xs[s].copy()
        pL0 = -ps[s].copy()
        xR = xs[s].copy()
        pR = ps[s].copy()

        # Backward evolve right system
        for _ in range(steps_back):
            xR, pR = leapfrog_step(xR, pR, dt_back, J, K, m)

        # Duplicate & perturb
        xR_dup = xR.copy()
        pR_dup = pR.copy()
        xR_dup[0] += pert

        # Forward evolve both to t=0
        for _ in range(returnto_step):
            xR_dup, pR_dup = leapfrog_step(xR_dup, pR_dup, dt, J, K, m)
            xR, pR = leapfrog_step(xR, pR, dt, J, K, m)

        # Coupling kick
        pL = pL0.copy()
        xL = xL0.copy()

        delta_x2 = (xR_dup[1] - xL[1])
        pL[1] += steps_g * delta_x2

        # Forward evolution and record
        for t in range(steps_forward):
            xL0, pL0 = leapfrog_step(xL0, pL0, dt, J, K, m)
            xL, pL = leapfrog_step(xL, pL, dt, J, K, m)

            delta_p1L[s, t] = pL[0] - pL0[0]

    poisson_bracket = delta_p1L / (pert * steps_g)

    # Compute statistics
    mean_delta = np.zeros(steps_forward)
    std_delta = np.zeros(steps_forward)

    for t in range(steps_forward):
        vals = delta_p1L[:, t]
        mean = np.mean(vals)
        var = np.mean(vals * vals) - mean * mean
        mean_delta[t] = mean
        std_delta[t] = np.sqrt(max(var, 0.0))


    return mean_delta, std_delta, poisson_bracket, delta_p1L


# Warm-up
simulation_loop(xs_sub[:1], ps_sub[:1], J_sub, K_sub,
                m, dt, 1, dt_back, 1,
                steps_g, pert, 1)

# Run simulation with timing
start_time = tm.time()

mean_delta_p1L, std_delta_p1L, poisson_bracket, delta_p1L = simulation_loop(
    xs_sub, ps_sub, J_sub, K_sub,
    m, dt,
    steps_back, dt_back,
    returnto_step, steps_g,
    pert, steps_forward
)

elapsed = tm.time() - start_time

# Plot
time_axis = np.arange(steps_forward) * dt

plt.figure(figsize=(7, 4))
plt.plot(time_axis, mean_delta_p1L)
plt.plot(time_axis, mean_delta_p1L + std_delta_p1L, 'k--', linewidth=0.8, alpha=0.5)
plt.plot(time_axis, mean_delta_p1L - std_delta_p1L, 'k--', linewidth=0.8, alpha=0.5)
plt.xlabel("Time after coupling")
plt.ylabel(r"$\delta p^1_L(t)$")
plt.title(
    f"Mean Momentum Difference of Particle 1\n"
    f"N = {N_subset}, Samples = {num_samples_subset}, "
    f"g = {steps_g}, Runtime = {elapsed:.2f}s, δx = {pert}"
)

plt.tight_layout()
plt.savefig("delta_p_compressed.pdf", dpi=300, bbox_inches='tight')
plt.show()

t_forward = np.arange(steps_forward) * dt
mean_pb = np.mean(poisson_bracket, axis=0)

plt.figure(figsize=(9, 5))
plt.plot(t_forward, mean_pb)
plt.xlabel('Time')
plt.ylabel(r'$\{p^1_R(t_R), p^1_L(t_L)\}$')
plt.title(
    f'Poisson Bracket (tR = {steps_back * dt:.3f})\n'
    f"N = {N_subset}, Samples = {num_samples_subset}, "
    f"g = {steps_g}, Runtime = {elapsed:.2f}s, δx = {pert}"
)
plt.axhline(0, color='k', linestyle='--', lw=0.8)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot individual sample trajectories
time_axis = np.arange(steps_forward) * dt

plt.figure(figsize=(10, 5))
for s in range(10):  # plot first 10 samples
    plt.plot(time_axis, delta_p1L[s, :], alpha=0.4, linewidth=0.8)

plt.xlabel("Time after coupling")
plt.ylabel("Δp (particle 1)")
plt.title("Individual sample trajectories of Δp¹L(t)")
plt.grid(True)
plt.show()

print(f"Time elapsed: {elapsed:.2f} s")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=time_axis, y=mean_delta_p1L, name='Mean δp'))
fig1.add_trace(go.Scatter(
    x=np.concatenate([time_axis, time_axis[::-1]]),
    y=np.concatenate([mean_delta_p1L + std_delta_p1L, (mean_delta_p1L - std_delta_p1L)[::-1]]),
    fill='toself', opacity=0.3, name='±1 std'
))
fig1.update_layout(
    xaxis_title="Time after coupling", yaxis_title="Mean δp (particle 1)"
)
fig1.write_html("plot1_mean_momentum.html")