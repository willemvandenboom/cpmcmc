import os

# For `cppyy` to find the Boost library:
os.environ["EXTRA_CLING_ARGS"] = "-I/usr/local/include "

import cppyy
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

import cpmcmc

np.seterr(divide="ignore")  # Allow for np.log(0.0)
cppyy.include("horseshoe.h")

# Reduce Python overhead:
coupled_sample_eta_cpp = cppyy.gbl.coupled_sample_eta_cpp
rng=np.random.default_rng(seed=0)

# Simulation set-up as in Section 3.1 of arXiv:2012.04798v1
n = 100
p = 20
s = 10
beta_star = np.zeros(p)

for j in range(s):
    beta_star[j] = 2.0**(0.25 * (8 - j))

W = rng.normal(size=(n, p))
sigma_star = 8.0
y = W@beta_star + rng.normal(scale=sigma_star, size=n)
I_n = np.eye(n)
omega = 1.0  # As in arXiv:2012.04798v1
s_MH = 0.8  # As in arXiv:2012.04798v1
N = 25  # Number of SMC particles
min_iter = 10**3  # Total of coupled and marginal iterations is at least this.
max_iter = 5 * min_iter  # Max. no. of iterations before giving up on coupling


def log_likelihood(x_i):
    N = len(x_i)
    res = np.empty(N)

    for i in range(N):
        res[i] = n*np.log(x_i[i][3]) + np.sum((y - W@x_i[i][0])**2)/x_i[i][3]

    return -0.5 * res


def h(x_i):
    """Univariate statistic being assessed by these simulations"""
    if type(x_i) is list:
        return x_i[0][s] + x_i[0][s]**2
    
    N = len(x_i)
    h_arr = np.empty(N)
    
    for i in range(N):
        h_arr[i] = x_i[i][0][s] + x_i[i][0][s]**2
    
    return h_arr


def sample_P(m, U, rng):
    T = 1.0/U - 1.0
    
    return -np.log1p(np.expm1(-m * T) * rng.random(
        size=m.shape if type(m) is np.ndarray else 1
    )) / m


def log_dens_P(eta, m, U):
    T = 1.0/U - 1.0
    
    if eta > T:
        return -np.inf
    
    return np.log(m) - m*eta - np.log(-np.expm1(-m * T))


def sample_eta(eta, m, rng):
    """
    Slice sampling for eta per Algorithm 2 of arXiv:2012.04798v1 with 𝜈 = 1
    """
    
    # Step 1
    U = rng.uniform(high=1.0 / (1.0 + eta))
    
    # Step 2
    return sample_P(m, U, rng)


def M_xi(xi, eta, α):
    tmp = W * np.sqrt(α / xi / eta)
    return I_n + tmp@tmp.T


def log_p_xi(xi, eta, α):
    """Log of numerator/denominator of acceptance probability"""
    M = M_xi(xi, eta, α)
    M_chol = scipy.linalg.lapack.dpotrf(a=M, lower=1, overwrite_a=1)[0]

    rate = 0.5 * (omega + α*np.sum(scipy.linalg.solve_triangular(
        a=M_chol, b=y, lower=True, check_finite=False
    )**2))
    
    try:
        accept = -np.log(np.diag(M_chol)).sum() - 0.5*(
            (α*n + omega)*np.log(rate) + np.log(xi)
        ) - np.log1p(xi) + np.log(xi)
    except:
        print("Error in evaluating acceptance probability.")
        accept = -np.inf

    return accept, rate, M_chol


def MCMC_update(x_i, rng, α=1.0):
    """MCMC algorithm per Algorithm 1 of arXiv:2012.04798v1 with 𝜈 = 1"""
    if type(x_i) is not list:
        for i in range(len(x_i)):
            x_i[i] = MCMC_update(x_i[i], rng, α)
        
        return x_i
    
    beta, eta, xi, sigma2 = x_i
    
    # Step 1
    # Sample eta
    eta = sample_eta(eta=eta, m=0.5 * xi / sigma2 * beta**2, rng=rng)

    # Step 2(a)
    # Sample xi
    xi_prop = min(rng.lognormal(mean=np.log(xi), sigma=s_MH), 1e100)
    accept, rate, M_chol = log_p_xi(xi, eta, α)
    accept_prop, rate_prop, M_chol_prop = log_p_xi(xi_prop, eta, α)

    if np.log(rng.random()) < accept_prop - accept:
        xi = xi_prop
        rate = rate_prop
        M_chol = M_chol_prop

    # Step 2(b)
    # Sample sigma2
    sigma2 = 1.0 / rng.gamma(shape=0.5 * (omega + α*n), scale = 1.0 / rate)

    # Step 2(c)
    # Sample beta
    # Efficient method as per Equation 5 of 
    # http://jmlr.org/papers/v21/19-536.html
    u = rng.normal(size=p) / np.sqrt(xi * eta)
    α_sqrt = np.sqrt(α)
    v = W@u*α_sqrt + rng.normal(size=n)
    sigma = np.sqrt(sigma2)

    v_star = scipy.linalg.cho_solve(
        c_and_lower=(M_chol, True), b=y*(α_sqrt / sigma) - v,
        overwrite_b=True, check_finite=False
    )

    beta = α_sqrt * sigma * (u + (W / (xi * eta)).T@v_star)

    return [beta, eta, xi, sigma2]


def sample_from_prior(rng):
    eta = rng.standard_cauchy(size=p)**2
    xi = min(rng.standard_cauchy()**2, 1e100)
    sigma2 = 1.0 / rng.gamma(shape=0.5 * omega, scale=2.0 / omega)
    beta = rng.normal(scale=np.sqrt(sigma2 / xi / eta))
    return [beta, eta, xi, sigma2]


def equal(z):
    """
    Check whether the two conditional SMC chains have met.
    
    This function also works for `z` from the non-SMC MCMC
    from arXiv:2012.04798v1.
    """
    if z.ndim == 1:
        return np.allclose(z[0][0], z[1][0]) \
            and np.allclose(z[0][1], z[1][1]) \
            and np.isclose(z[0][2], z[1][2]) and np.isclose(z[0][3], z[1][3])
    
    for s in range(z.shape[1]):
        if not equal(z[:, s]):
            return False
    
    return True


def maximal_coupling(P, Q, log_p, log_q, rng):
    """
    Maximal coupling with independent residuals
    (Algorithm 7 of arXiv:2012.04798v1)
    """
    X = P(rng)
    W = rng.random()
    
    if log_p(X) + np.log(W) <= log_q(X):
        return X, X
    
    while True:
        Y_tilde = Q(rng)
        W_tilde = rng.random()
        
        if log_q(Y_tilde) + np.log(W_tilde) > log_p(Y_tilde):
            return X, Y_tilde


def log1mexp(arg):
    """
    log1mexp(arg) = log(1 - exp(-arg)) for arg > 0.

    Evaluate log(1 - exp(-arg)) per Equation 7 from
    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf.
    """
    return np.where(
        arg < 0.693,
        np.log(-np.expm1(-arg)),
        np.log1p(-np.exp(-arg))
    )


def coupled_sample_eta(eta_both, m_both, rng):
    """
    Coupled slice sampling for eta per Algorithm 4 of arXiv:2012.04798v1 with
    𝜈 = 1
    """

    coupled_sample_eta_cpp(
        eta_both[0], eta_both[1], m_both[0], m_both[1], p,
        cpmcmc.random_seed(rng)
    )

#     # Step 1
#     U_crn = rng.random(p)
#     U_both = np.empty(2, dtype=object)

#     for j in range(2):
#         U_both[j] = U_crn / (1.0 + eta_both[j])

#     # Step 2
#     for j in range(p):
#         eta_both[0][j], eta_both[1][j] = maximal_coupling(
#             P=lambda rng: sample_P(m_both[0][j], U_both[0][j], rng),
#             Q=lambda rng: sample_P(m_both[1][j], U_both[1][j], rng),
#             log_p=lambda eta: log_dens_P(eta, m_both[0][j], U_both[0][j]),
#             log_q=lambda eta: log_dens_P(eta, m_both[1][j], U_both[1][j]),
#             rng=rng
#         )

    return eta_both


def coupled_MCMC_update(z_i, rng, α=1.0):
    """
    The two-scale coupled MCMC algorithm from Algorithm 5 of arXiv:2012.04798v1
    with 𝜈 = 1
    """
    if z_i.ndim == 2:
        for i in range(len(z_i)):
            z_i[i, :] = coupled_MCMC_update(z_i[i, :], rng, α)
        
        return z_i
    
    if equal(z_i):
        z_i[0] = MCMC_update(z_i[0], rng, α)
        z_i[1] = z_i[0]
        return z_i
    
    # Step 1
    # Sample eta
    m_both = [0.5 * z_i[j][2] / z_i[j][3] * z_i[j][0]**2 for j in range(2)]

    # Choose between the "scale" of the coupling as in arXiv:2012.04798v1.
    # Code follows Proposition B.2 in arXiv:2012.04798v1 with 𝜈 = 1.
    # Another implementation of this is at
    # https://github.com/niloyb/CoupledHalfT/blob/c7f1e4db7270e15661217a54f4383c5d548b4bb0/R/eta_half_t_sampling.R#L253-L280
    # with `t_dist_df = 1` and `iterations = 1`.
    crn_unif = rng.random(size=p)
    T = [(1.0 + z_i[j][1])/crn_unif - 1.0 for j in range(2)]
    mT = [m_both[j] * T[j] for j in range(2)]
    T_min = np.minimum(T[0], T[1])
    ind = m_both[0] == m_both[1]
    log_prob = np.empty(p)

    log_prob[ind] = log1mexp(m_both[0][ind] * T_min[ind]) - log1mexp(
        m_both[0][ind] * np.maximum(T[0][ind], T[1][ind])
    )

    K_tilde = np.minimum(
        np.maximum(0.0, (
            np.log(m_both[1]) - np.log(m_both[0]) \
                + log1mexp(mT[0]) - log1mexp(mT[1])
        ) / np.where(ind, 1.0, m_both[1] - m_both[0])),
        T_min
    )

    m_min = np.minimum(m_both[0], m_both[1])
    m_max = np.maximum(m_both[0], m_both[1])
    T_m_min = np.empty(p)
    T_m_max = np.empty(p)

    for j in range(2):
        tmp = m_both[j] < m_both[1 - j]
        T_m_min[tmp] = T[j][tmp]
        T_m_max[tmp] = T[1 - j][tmp]

    ind2 = np.logical_not(ind)

    log_prob[ind2] = np.log(np.exp(
        log1mexp(m_min[ind2] * K_tilde[ind2]) \
            - log1mexp(m_min[ind2] * T_m_min[ind2])
    ) + np.exp(np.log(
        np.exp(log1mexp(m_max[ind2] * T_min[ind2])) \
            - np.exp(log1mexp(m_max[ind2] * K_tilde[ind2]))
    ) - log1mexp(m_max[ind2] * T_m_max[ind2])))

    # We use a threshold of 0.5 as in arXiv:2012.04798v1.
    if 1.0 - np.exp(np.sum(log_prob)) < 0.5:
        z_i[0][1], z_i[1][1] = coupled_sample_eta(
            eta_both=[z_i[j][1] for j in range(2)], m_both=m_both, rng=rng
        )
    else:
        # Common random numbers update for eta
        # Algorithm 6 in arXiv:2012.04798v1

        # Step 1
        W_crn = rng.uniform(size=p)

        # Step 2
        V_crn = rng.uniform(size=p)

        for j in range(2):
            T = (1.0 + z_i[j][1])/W_crn - 1.0
            z_i[j][1] = -np.log1p(np.expm1(-m_both[j] * T) * V_crn) / m_both[j]

    # Step 2(a)
    # Coupled Metropolis-Hastings for xi
    # Algorithm 11 of arXiv:2012.04798v1
    xi_prop_both = np.minimum(np.exp(maximal_coupling(
        P=lambda rng: rng.normal(loc=np.log(z_i[0][2]), scale=s_MH),
        Q=lambda rng: rng.normal(loc=np.log(z_i[1][2]), scale=s_MH),
        log_p=lambda t: -0.5 * (t - np.log(z_i[0][2]))**2 / s_MH,
        log_q=lambda t: -0.5 * (t - np.log(z_i[1][2]))**2 / s_MH,
        rng=rng
    )), 1e100)

    tmp = [log_p_xi(z_i[j][2], z_i[j][1], α) for j in range(2)]
    tmp_prop = [log_p_xi(xi_prop_both[j], z_i[j][1], α) for j in range(2)]
    rate_both = [tmp[j][1] for j in range(2)]
    M_chol_both = [tmp[j][2] for j in range(2)]
    U_star_log = np.log(rng.random())

    for j in range(2):
        if U_star_log < tmp_prop[j][0] - tmp[j][0]:
            z_i[j][2] = xi_prop_both[j]
            rate_both[j] = tmp_prop[j][1]
            M_chol_both[j] = tmp_prop[j][2]

    # Step 2(b)
    # Sample sigma2 using a maximal coupling    
    a=0.5 * (omega + α*n)

    z_i[0][3], z_i[1][3] = maximal_coupling(
        P=lambda rng: 1.0 / rng.gamma(shape=a, scale = 1.0 / rate_both[0]),
        Q=lambda rng: 1.0 / rng.gamma(shape=a, scale = 1.0 / rate_both[1]),
        log_p=lambda sigma2: a*np.log(rate_both[0]) \
            + (-a - 1.0)*np.log(sigma2) - rate_both[0]/sigma2,
        log_q=lambda sigma2: a*np.log(rate_both[1]) \
            + (-a - 1.0)*np.log(sigma2) - rate_both[1]/sigma2,
        rng=rng
    )

    # Step 2(c)
    # Sample beta using a common random numbers coupling
    # Algorithm 10 of arXiv:2012.04798v1
    α_sqrt = np.sqrt(α)
    r = rng.normal(size=p)
    delta = rng.normal(size=n)
    u_both = [r / np.sqrt(z_i[j][2] * z_i[j][1]) for j in range(2)]
    v_both = [W@u_both[j]*α_sqrt + delta for j in range(2)]
    sigma_both = [np.sqrt(z_i[j][3]) for j in range(2)]

    v_star_both = [scipy.linalg.cho_solve(
        c_and_lower=(M_chol_both[j], True),
        b=y*(α_sqrt / sigma_both[j]) - v_both[j],
        overwrite_b=True, check_finite=False
    ) for j in range(2)]

    for j in range(2):
        z_i[j][0] = α_sqrt * sigma_both[j] * (
            u_both[j] + (W / (z_i[j][2] * z_i[j][1])).T@v_star_both[j]
        )

    return z_i


def test_func(x_i):
    N = len(x_i)
    ret = np.empty(N, dtype=int)
    
    for i in range(N):
        ret[i] = np.sum(abs(x_i[i][0]) > 0.01)
    
    return ret


setup = cpmcmc.adapt_SMC(
    10**4, sample_from_prior, log_likelihood, MCMC_update, test_func, rng=rng
)


def coupled_MCMC_Biswas(rng=np.random.default_rng()):
    start_time = cpmcmc.time.perf_counter()
    z = np.empty(2, dtype=object)
    z[0] = sample_from_prior(rng)
    z[1] = z[0]
    z[0] = MCMC_update(z[0], rng)
    meeting_time = 0
    h_coupled = [[h(z[j]) for j in range(2)]]

    for ii in range(max_iter):
        coupled = equal(z)
        
        if coupled:
            break
        
        z = coupled_MCMC_update(z, rng)
        h_coupled.append([h(z[j]) for j in range(2)])
        meeting_time += 1

        if meeting_time % 500 == 0:
            print("Biswas: Not yet met after", meeting_time, "steps.")
    
    meet_time = cpmcmc.time.perf_counter()
    meeting_time = ii + 1
    
    if not coupled:
        warnings.warn("Failed to meet: The estimator is biased!")
    
    x = z[0]
    
    for _ in range(meeting_time, min_iter):
        x = MCMC_update(x, rng)
        h_coupled.append([h(x), None])
    
    end_time = cpmcmc.time.perf_counter()

    return {
        "meeting time": meeting_time,
        "h coupled": np.array(h_coupled, dtype=float),
        "meet time": meet_time - start_time,
        "additional time": end_time - meet_time
    }


S = 128

result_list = [cpmcmc._repeat_cMCMC(S=S, pyfunc=coupled_MCMC_Biswas, rng=rng)]

for PIMH_prob in [1.0, 0.9]:
    result_list.append(cpmcmc.repeat_cMCMC(
        S, min_iter, max_iter, equal, N, sample_from_prior,
        log_likelihood, MCMC_update, setup, coupled_MCMC_update, h,
        rng=rng, PIMH_prob=PIMH_prob
    ))


n_bootstrap = 10**3
alpha_level = 0.3
col = "black"
n_setup = len(result_list)
h_est_all = n_setup * [None]


def plot_result(
    result, ax, l_m=10, prob=None, label=None, color="black", h_est_list=None
):
    compute_h = h_est_list is None or l_m != len(h_est_list)
    
    if compute_h:
        h_est_list = l_m * [None]
    
    if prob is not None:
        label = "Prob. of PIMH: " + str(prob)
    
    l_list = np.exp(np.linspace(
        start=0.0, stop=np.log(min_iter), num=l_m
    )).round().astype(int)

    var_arr = np.empty((l_m, 3))
    time_arr = np.empty(l_m)
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
                time_tmp[s] += (l - tau) * result["time"][s, 1] \
                    / (min_iter - tau)
            
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

    parts = axs[ind_tau, i].violinplot(
        np.log10(result_list[i]["meeting time"])
    )

    axs[2, i].set_xscale("log")
    axs[2, i].set_xticks(10**np.arange(0, np.floor(np.log10(min_iter)) + 1))
    
    for key in ["cmins", "cmaxes", "cbars"]:
        parts[key].set_edgecolor(col)
    
    h_est_all[i] = plot_result(
        result_list[i], ax=axs[2, i], l_m=100, h_est_list=h_est_all[i]
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
        "Biswas et al. (2020)",
        "PIMH",
        "Mixed",
    ][ind])

fig.savefig("horseshoe.pdf")