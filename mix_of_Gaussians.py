import cppyy
import matplotlib.pyplot as plt
import numpy as np

import cpmcmc

rng = np.random.default_rng(seed=0)

# Simulation as per Section B.2 of Middleton et al. (2019).
x_star = np.array([-3.0, 0.0])
D = len(x_star)
M = 100
sigma = 1.0

"""
The cluster allocation in the next line is not drawn randomly according to the
model but instead follows what was done in Middleton et al. (2019) per
https://github.com/particlemontecarlo/unbiased_pimh/blob/f89d7b0210ad5ae98d21c4c6cffad4df22b4e1e1/R/model_get_smc.R
"""
y = rng.normal(loc=x_star[np.arange(M) % D], scale=sigma)


N = 25 # Number of SMC particles
min_iter = 10**4  # Total of coupled and marginal iterations is at least this.
max_iter = min_iter # Maximum number of iterations before giving up on coupling

if sigma != 1.0:
    raise NotImplementedError("The following C++ code assumes sigma = 1.0.")


cppyy.cppdef("""
void log_likelihood_cpp(double* vec_out, double* x, double* y, int N) {
    /*
    Log-likelihood with constants dropped
    
    This function vectorizes over the number of particles `N`.
    That is, it assumes that `x` is a `N` by `D` matrix and outputs
    to the `N`-dimensional vector `vec_out`.
    */
    int ind = 0;
    
    for (int i = 0; i < N; i++) {
        double term1 = 0.0;
        double term2 = 0.0;
        
        for (int j = 0; j < """ + str(D) + """; j++) {
            double x_ij = x[ind++];
            term2 += x_ij * x_ij;
            
            for (int m = 0; m < """ + str(M) + """; m++) {
                term1 += x_ij * y[m];
            }
        }
        
        vec_out[i] = term1 - 0.5*""" + str(M) + """*term2;
    }
}
""")


# Reduce Python overhead:
log_likelihood_cpp = cppyy.gbl.log_likelihood_cpp


def log_likelihood(x_i):
    """
    Log-likelihood with constants dropped
    """
    N = len(x_i)
    x_arr = np.stack(x_i) if x_i.ndim == 1 else x_i
    x_arr = np.ascontiguousarray(x_arr)
    ret = np.empty(N)
    log_likelihood_cpp(ret, x_arr, y, N)
    return ret


def h(x_i):
    """Same statistic as in Section B.2 of Middleton et al. (2019)"""
    x_arr = np.stack(x_i)
    return x_arr[:, 0] + x_arr[:, 1] + x_arr[:, 0]**2 + x_arr[:, 1]**2


def MCMC_update(x_i, rng, α=1.0):
    """
    MCMC update for each `x` in the 1D NumPy array `x_i`
    
    Metropolis-Hastings algorithm with as proposal a Gaussian centered at `x`
    with identity covariance. Thus, the proposal is a symmetric random walk.
    """
    N = len(x_i)
    x_arr = np.stack(x_i)
    x_prop = x_arr + rng.normal(size=(N, D))
    
    accept_mask = (
        np.log(rng.random(size=N)) < α * \
            (log_likelihood(x_prop) - log_likelihood(x_arr))
    ) & (x_prop.min(axis=-1) > -10.0) & (x_prop.max(axis=-1) < 10.0)
    
    for i in accept_mask.nonzero()[0]:
        x_i[i] = x_prop[i, :]
    
    return x_i


def sample_from_prior(rng):
    return rng.uniform(low=-10.0, high=10.0, size=D)


def equal(z):
    return np.allclose(np.stack(z[0, :]), np.stack(z[1, :]))


def coupled_MCMC_update(z_i, rng, α=1.0):
    # Common seed MCMC update
    tmp_seed = cpmcmc.random_seed(rng)
    
    for j in range(2):
        z_i[:, j] = MCMC_update(
            x_i=z_i[:, j], rng=np.random.default_rng(tmp_seed), α=α
        )
    
    return z_i


def test_func(x_i):
    return np.linalg.norm(np.stack(x_i), axis=1)


setup = cpmcmc.adapt_SMC(
    10**4, sample_from_prior, log_likelihood, MCMC_update, test_func, rng=rng
)

S = 1024

result_list = [cpmcmc.repeat_cMCMC(
    S, min_iter, max_iter, equal, N, sample_from_prior, log_likelihood,
    MCMC_update, setup, coupled_MCMC_update, h, rng=rng, PIMH_prob=prob
) for prob in [0.0, 0.5, 1.0]]

n_bootstrap = 10**3
alpha_level = 0.3
col = "black"
n_setup = len(result_list)
h_est_all = n_setup * [None]


def plot_result(
    result, ax, n_l=10, prob=None, label=None, color="black", h_est_list=None
):
    compute_h = h_est_list is None or n_l != len(h_est_list)
    
    if compute_h:
        h_est_list = n_l * [None]
    
    if prob is not None:
        label = "Prob. of PIMH: " + str(prob)
    
    l_list = np.exp(np.linspace(
        start=0.0, stop=np.log(min_iter), num=n_l
    )).round().astype(int)

    var_arr = np.empty((n_l, 3))
    time_arr = np.empty(n_l)
    S = len(result["meeting time"])
    
    if compute_h:
        h_arr = np.full(
            shape=(S, max(min_iter, max_iter), 2), fill_value=np.nan
        )

        for s in range(S):
            h_coupled = result["h"][s]
            tau = result["meeting time"][s]

            for t in range(tau):
                h_arr[s, t, :] = h_coupled[t]

            for t in range(tau, min_iter):
                h_arr[s, t, 0] = h_coupled[t][0]

        h_diff = -np.diff(h_arr)[:, :, 0]
    
    for ind, l in enumerate(l_list):
        if compute_h:
            # Ergodic average (EA) and bias correction (BC) for each k
            EA_arr = (
                h_arr[:, :l, 0][:, ::-1].cumsum(axis=1) / np.arange(1, l + 1)
            )[:, ::-1]

            BC_arr = np.empty((S, l))

            for k in range(l):
                for s in range(S):
                    BC_arr[s, k] = np.sum(np.minimum(
                        l - k, np.arange(k + 1, result["meeting time"][s]) - k
                    ) * h_diff[s, (k + 1):result["meeting time"][s]])

                BC_arr[:, k] /= l - k

            h_est = EA_arr + BC_arr
            h_est_list[ind] = h_est
        else:
            h_est = h_est_list[ind]
        
        var_tmp = h_est.var(axis=0)
        k = var_tmp.argmin()
        var_arr[ind, 0] = var_tmp[k]

        # Bootstrapping to get 2.5th and 97.5th percentiles
        var_arr[ind, 1:] = np.percentile(a=rng.choice(
            a=h_est[:, k], size=(n_bootstrap, S), replace=True
        ).var(axis=1), q=[2.5, 97.5])
        
        time_tmp = np.empty(S)
        
        for s in range(S):
            time_tmp[s] = result["time"][s, 0]
            tau = result["meeting time"][s]
            
            if l > tau:
                time_tmp[s] += (l - tau) * result["time"][s, 1] / (
                    min_iter - tau
                )
            
        time_arr[ind] = time_tmp.mean()

    ax.plot(l_list, time_arr * var_arr[:, 0], label=label, color=color)

    ax.fill_between(
        l_list, time_arr * var_arr[:, 1], time_arr * var_arr[:, 2],
        alpha=alpha_level, facecolor=color
    )

    ax.set_xscale("log")    
    ax.set_yscale("log")
    ax.set_xlabel(r"$l$")
    return h_est_list


fig, axs = plt.subplots(
    3, n_setup, figsize=(8, 8), sharex="row", sharey="row",
    gridspec_kw={'hspace': 0, 'wspace': 0}
)

ind_tau = 0
ind_IAT = 1

if n_setup == 1:
    axs = axs[:, np.newaxis]

for i in range(n_setup):
    IAT = cpmcmc.get_IAT(result_list[i], burnin=min_iter // 2)
    
    # Bootstrapping to get 2.5th and 97.5th percentiles
    CI = np.percentile(a=rng.choice(
        a=IAT, size=(n_bootstrap, S), replace=True
    ).mean(axis=1), q=[2.5, 97.5])
    
    CI_width = 15

    axs[ind_IAT, i].vlines(
        x=0.0, ymin=CI[0], ymax=CI[1], color=col, alpha=alpha_level,
        lw=CI_width
    )

    axs[ind_IAT, i].scatter(
        x=0.0, y=IAT.mean(), s=CI_width**2, marker="_", color=col
    )

    parts = axs[ind_tau, i].violinplot(np.log10(
        result_list[i]["meeting time"]
    ))

    axs[2, i].set_xscale("log")
    axs[2, i].set_xticks(10**np.arange(0, np.floor(np.log10(min_iter)) + 1))
    
    for key in ["cmins", "cmaxes", "cbars"]:
        parts[key].set_edgecolor(col)
    
    h_est_all[i] = plot_result(
        result_list[i], ax=axs[2, i], n_l=100, h_est_list=h_est_all[i]
    )

    parts["bodies"][0].set_alpha(alpha_level)
    parts["bodies"][0].set_facecolor(col)

axs[ind_IAT, 0].set_ylabel("Integrated autocorrelation time")
axs[ind_IAT, 0].set_xticks([])
n_ticks = np.round(axs[ind_tau, 0].get_ylim()[1])
axs[ind_tau, 0].set_yticks(ticks=np.arange(n_ticks))
axs[ind_tau, 0].set_yticklabels(labels=10**np.arange(n_ticks, dtype=int))
axs[ind_tau, 0].set_ylabel(r"Meeting time $\tau$")

axs[2, 0].set_ylabel(
    r"$\hat{\mathrm{var}}(\bar{h}_k^l) \times$ time (seconds)"
)

for ind, ax in enumerate(axs[0, :]):
    ax.set_title([
        "Conditional SMC",
        "Mixed",
        "PIMH",
    ][ind])

axs[2, 0].set_ylim((7.0 * 10**-4, 3.0 * 10**-2))
fig.savefig("mix_of_Gaussians.pdf")