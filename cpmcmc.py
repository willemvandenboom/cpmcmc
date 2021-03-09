""""
Coupled particle Markov chain Monte Carlo

This Python module contains functions to simulate coupling of Markov chains
generated by iterated sequential Monte Carlo (SMC), also known as particle
Markov chain Monte Carlo (MCMC). Such coupled chains can then be used for
unbiased Monte Carlo estimation.
"""

import os
import time
import warnings

# See
# https://stackoverflow.com/questions/58909525/what-is-numpy-core-multiarray-umath-implement-array-function-and-why-it-costs-l
# Speedup by disabling additional NumPy array functionality:
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

# To prevent spawning of more than 1 thread: Actually faster in this case!
os.environ["MKL_NUM_THREADS"] = "1"

import cppyy
import IPython.core.magics.execution
import numpy as np
import p_tqdm
import rpy2.robjects.numpy2ri
import rpy2.robjects.packages as rpackages
import scipy.optimize

warnings.simplefilter("always", category=UserWarning)
large_int = 2**63


def random_seed(rng):
    # The following is faster than `rng.integers(large_int)`.
    return int(large_int * rng.random())


def sample(prob, uniform_samples):
    """
    Sample `size` integers from 0, 1, ..., `len(prob)`
    with probability vector `prob`.

    The standard uniform samples in `uniform_samples` are used.
    This function is faster than
    `rng.choice(a=len(prob), size=len(uniform_samples), p=prob)`.
    """
    return np.searchsorted(a=np.cumsum(prob), v=uniform_samples)


def coupled_sampling(p1, p2, n=None, rng=np.random.default_rng()):
    """
    Coupled sampling from the discrete distributions `p1` and `p2`.

    Algorithm 1 from Jasra et al. (2017, doi:10.1137/17M1111553):
    Sample an index from `p1` and `p2` such that the probability
    of the indices being equal is maximized.

    `n` is the number of samples.
    Its default is the length of `p1`.
    """
    dim = len(p1)

    if n is None:
        n = dim

    p_min = np.minimum(p1, p2)
    p_min_sum = p_min.sum()
    coupled = rng.random(size=n) < p_min_sum
    coupled_n = coupled.sum()
    res = np.empty((2, n), dtype=int)

    if coupled_n > 0:
        res[:, coupled.nonzero()] = sample(
            prob=p_min / p_min_sum,
            uniform_samples=rng.random(size=coupled_n),
        )

    if coupled_n < n:
        uniform_samples = rng.random(size=n - coupled_n)
        index = np.nonzero(~coupled)

        res[0, index] = sample(
            prob=(p1 - p_min) / (1.0 - p_min_sum),
            uniform_samples=uniform_samples
        )

        res[1, index] = sample(
            prob=(p2 - p_min) / (1.0 - p_min_sum),
            uniform_samples=uniform_samples
        )

    return res


cppyy.cppdef("""
void sys_resampling_cpp(long* vec_out, double* W, double U, int dim) {
    /*
    Systemic resampling
    
    This is Algorithm 3 of Chopin & Singh (doi:10.3150/14-BEJ629).
    The weights `W` should be normalized.
    `dim` is the number of weights.
    */
    double nu = dim * W[0];
    long m = 0;
    
    for (int n = 0; n < dim; n++) {
        while (nu < U) nu += dim * W[++m];
        vec_out[n] = m;
        U++;
    }
}
""")


# Reduce Python overhead:
sys_resampling_cpp = cppyy.gbl.sys_resampling_cpp


def sys_resampling(w, U):
    """
    Systemic resampling
    
    This is Algorithm 3 of Chopin & Singh (doi:10.3150/14-BEJ629).
    The weights `w` should be normalized.
    """
    tmp = np.ascontiguousarray(w)
    dim = len(tmp)
    a = np.empty(dim, dtype=int)
    sys_resampling_cpp(a, tmp, U, dim)
    return a
    
#     dim = len(w)
#     nu = dim * np.cumsum(w)
#     a = np.empty(dim, dtype=int)
#     m = 0
    
#     for n in range(dim):
#         while nu[m] < U - np.finfo(float).eps:
#             m += 1
        
#         a[n] = m
#         U += 1.0
    
#     return a


def cond_sys_resampling(W, rng):
    """
    Conditional systemic resampling
    
    This is Algorithm 4 of Chopin & Singh (doi:10.3150/14-BEJ629).
    """
    dim = len(W)
    nW1 = dim * W[0]
    
    # Step (a)
    if nW1 <= 1.0:
        U = nW1 * rng.random()
    else:
        r1 = nW1 % 1
        
        if rng.random() < r1 * np.ceil(nW1) / nW1:
            U = r1 * rng.random()
        else:
            U = r1 + (1.0 - r1)*rng.random()
    
    # Step (b)
    a_bar = sys_resampling(W, U)
    
    # Step (c)
    zero_loc = np.nonzero(a_bar == 0)
    n_zero = len(zero_loc)
    
    if n_zero == 1:
        return a_bar
    
    return np.roll(a_bar, -zero_loc[int(n_zero * rng.random())])


def coupled_cond_sys_resampling(W_both, rng):
    """
    Coupled conditional systemic resampling
    
    Marginally, this performs Algorithm 4 of
    Chopin & Singh (doi:10.3150/14-BEJ629).
    """
    dim = len(W_both)    
    nW1_both = dim * W_both[0, :]
    
    tmp = rng.random()
    tmp2 = rng.random()
    a_bar_both = np.empty((dim, 2), dtype=int)
    
    for j in range(2):
        # Step (a)
        if nW1_both[j] <= 1.0:
            U = nW1_both[j] * tmp
        else:
            r1 = nW1_both[j] % 1

            if tmp2 < r1 * np.ceil(nW1_both[j]) / nW1_both[j]:
                U = r1 * tmp
            else:
                U = r1 + (1.0 - r1)*tmp
    
        # Step (b)
        a_bar_both[:, j] = sys_resampling(W_both[:, j], U)
    
    # Step (c)
    cycles = [np.nonzero(a_bar_both[:, j] == 0) for j in range(2)]
    n_cycles = [len(cycles[j]) for j in range(2)]
    overlap = np.empty(n_cycles, dtype=int)
    
    def to_cycle(j, i):
        return np.roll(a_bar_both[:, j], -cycles[j][i])
    
    for i0 in range(n_cycles[0]):
        cycle = to_cycle(0, i0)
        
        for i1 in range(n_cycles[1]):
            overlap[i0, i1] = np.sum(cycle == to_cycle(1, i1))
    
    coupling = np.zeros(n_cycles)
    ratio = n_cycles[0] / n_cycles[1]
    
    # Greedy coupling. Better choices might be available, e.g.
    # https://en.wikipedia.org/wiki/Hungarian_algorithm in case `overlap`
    # is a square matrix.
    #while coupling.sum() < dim - np.finfo(float).eps:
    for _ in range(n_cycles[0] * n_cycles[1]):
        ind = np.unravel_index(overlap.argmax(), shape=n_cycles)
        overlap[ind] = 0
        coupling[ind] = min(ratio, 1.0 - coupling[ind[0], :].sum())
        
    ind = np.unravel_index(
        np.min(coupling.cumsum() < rng.random()), shape=n_cycles
    )

    return np.array([to_cycle(j, ind[j]) for j in range(2)])


def adapt_SMC(
    N, sample_from_prior, log_likelihood, MCMC_update, test_func=None,
    S_eff_prop=0.8, rng=np.random.default_rng(), log_likelihood_max=None
):
    """
    Adapt the SMC sampler

    Adapt the tempering constants `α` and the number of MCMC steps for SMC.
    The adaptation is based on `N` particles.
    
    `sample_from_prior` is a Python function that takes an `np.Generator`
    instance as paramter `rng` and outputs an object `x` that is drawn from the
    prior.
    
    `log_likelihood` is a Python function that takes a 1-dimensional NumPy
    array of objects `x` as input and outputs a NumPy array of numbers.
    
    `MCMC_update` is a Python function that takes a 1-dimensional NumPy
    array of objects `x_i` as input and outputs a NumPy array of objects of the
    same length. Additionally, it has as parameters the `np.Generator` instance
    `rng` and the tempering temperature `α`.

    `test_func` is an optional Python function that takes a 1-dimensional NumPy
    array of objects `x` as input and outputs a NumPy array of floats. It is
    used to determine the number of MCMC steps.

    The optional argument `log_likelihood_max` is an upper bound to the
    likelihood. If it is provided, then the zeroth temperature is adapted using
    rejection sampling.
    """
    MCMC_steps = np.empty(0, dtype=int)
    S_eff_target = N / S_eff_prop if S_eff_prop > 1.0 else N * S_eff_prop
    max_m = 10**3
    max_MCMC = 10**3

    # Initialize time s = 0.
    if log_likelihood_max is None:
        α_s = np.zeros(1)
        x_i = np.empty(N, dtype=object)

        for i in range(N):
            x_i[i] = sample_from_prior(rng=rng)
    else:
        N_prop = 50 * N
        x_prop = np.empty(N_prop, dtype=object)
        print("Adapting the rejection sampler...")
        
        for i in range(N_prop):
            x_prop[i] = sample_from_prior(rng=rng)
        
        # Base first temperature on accepting at least `N` proposals.
        tmp = np.log(rng.random(size=N_prop)) / (
            log_likelihood(x_prop) - log_likelihood_max
        )
        
        α_s = np.array([np.sort(tmp)[-N]])
        
        if α_s[0] >= 1.0:
            raise NotImplementedError(
                "No SMC required: Rejection sampling suffices."
            )
        
        x_i = x_prop[tmp >= α_s[0]]

    for s in range(max_m):
        print("Time step", s, end="\r")
        w_log = log_likelihood(x_i)
        w_log -= np.max(w_log)
        α_diff_max = 1.0 - α_s[-1]

        # Determine α based on the effective sample size
        def weights(α_diff):
            w = np.exp(α_diff * w_log)
            return w / w.sum()

        def S_eff(α_diff):
            w = weights(α_diff)
            return 1/np.sum(w * w) - S_eff_target

        if S_eff(α_diff_max) >= 0:
            print("Number of SMC time steps:", len(α_s) - 1)
            print(α_s)
            print(MCMC_steps)
            return np.append(α_s, 1.0), MCMC_steps

        α_diff = scipy.optimize.root_scalar(
            S_eff, bracket=[0, α_diff_max]
        ).root

        α_s = np.append(α_s, α_s[-1] + α_diff)

        if s == max_m - 1:
            warnings.warn("Failed to reach `α = 1`.")
            return α_s, MCMC_steps

        W = weights(α_diff)
        x_i = x_i[sample(prob=W, uniform_samples=rng.random(size=N))]

        # Number of MCMC steps is determined per Equation 22 of Kantas et al.
        # (2013, arXiv:1307.6127v1).
        LL_i = α_s[-1] * log_likelihood(x_i)
        J_denom_LL = 2 * np.sum((LL_i - LL_i.mean())**2)

        if test_func is not None:
            test_i = test_func(x_i)
            J_denom_test = 2 * np.sum((test_i - test_i.mean())**2)

        for count_MCMC in range(max_MCMC):
            print("Time step", s, "MCMC step", count_MCMC, end="\r")
            x_i = MCMC_update(x_i=x_i, rng=rng, α=α_s[-1])  # MCMC step

            if np.sum(
                (α_s[-1]*log_likelihood(x_i) - LL_i)**2
            ) / J_denom_LL > 0.05:
                if test_func is None:
                    break

                if J_denom_test == 0.0:
                    warnings.warn(
                        "No diversity in `test_func` among particles." \
                        + "Unable to adapt based on `test_func`."
                    )

                    break

                if np.sum((test_func(x_i) - test_i)**2) / J_denom_test > 0.05:
                    break

        if count_MCMC == max_MCMC - 1:
            warnings.warn(
                "Failed to mutate particles sufficiently using MCMC."
            )

        MCMC_steps = np.append(MCMC_steps, count_MCMC + 1)


def resample(W, resample_threshold):
    if resample_threshold == 1.0:
        return True

    return 1.0 <= resample_threshold * len(W) * np.sum(W * W)



def get_weights(s, x_i, W, log_Z, α_s, log_likelihood):
    w_log = (α_s[s + 1] - α_s[s]) * log_likelihood(x_i[:, s])
    w_log_max = w_log.max()
    w_log -= w_log_max
    W *= np.exp(w_log)
    W_sum = W.sum()
    # Equation 14 of
    # Del Moral et al. (2006, doi:10.1111/j.1467-9868.2006.00553.x):
    log_Z += np.log(W_sum) + w_log_max
    return W / W_sum, log_Z


def default_h(x_i):
    return np.zeros(len(x_i))


def SMC(
    N, sample_from_prior, log_likelihood, MCMC_update, setup, h=default_h,
    systematic_resampling=True, resample_threshold=0.5,
    rng=np.random.default_rng()
):
    """
    Run SMC with `N` particles

    `sample_from_prior` is a Python function that takes an `np.Generator`
    instance as parameter `rng` and outputs an object `x` that is drawn from
    the prior if `α_s[0] == 0.0`.
    If `α_s[0] > 0.0`, then `sample_from_prior` instead samples from the first
    tempered posterior, and has the parameters `N` for the number of particles
    generate, temperature `α` and `np.Generator` object `rng`.
    
    `log_likelihood` is a Python function that takes a 1-dimensional NumPy
    array of objects `x` as input and outputs a NumPy array of numbers.

    `MCMC_update` is a Python function that takes a 1-dimensional NumPy
    array of objects `x_i` as input and outputs a NumPy array of objects of the
    same length. Additionally, it has as parameters the `np.Generator` instance
    `rng` and the tempering temperature `α`.

    `setup` is a tuple which defines the tempering and number of MCMC steps.

    `h` is a Python function that takes a 1-dimensional NumPy array of objects
    `x_i` as input and outputs a NumPy array of floats. It is the statistic one
    aims to obtain an unbiased estimate of.

    `systematic_resampling` specifies whether systematic or multinomial
    resampling is used.

    `resample_threshold` specifies the effective sample size as a proportion of
    `N` below which resampling happens. Resampling happens at each temperature
    if `resample_threshold == 1.0`. That is, there is no adaptive resampling in
    that scenario.
    """
    α_s, MCMC_steps = setup
    m = len(α_s) - 1

    # Initialize time s = 0.
    x_i = np.empty((N, m), dtype=object)

    if α_s[0] == 0.0:
        log_Z = 0.0

        for i in range(N):
            x_i[i, 0] = sample_from_prior(rng)
    else:
        x_i[:, 0] = sample_from_prior(N, α_s[0], rng)
        w_log = α_s[0] * log_likelihood(x_i[:, 0])
        w_log_max = w_log.max()
        w_log -= w_log_max
        W = np.exp(w_log)
        W_sum = W.sum() / N
        log_Z = np.log(W_sum) + w_log_max

    W = np.full(shape=N, fill_value=1.0 / N)

    for s in range(m):
        W, log_Z = get_weights(s, x_i, W, log_Z, α_s, log_likelihood)

        if s == m - 1:
            break
        
        if resample(W, resample_threshold):
            if systematic_resampling:
                a = sys_resampling(w=W, U=rng.random())
            else:
                a = sample(prob=W, uniform_samples=rng.random(size=N))
            
            x_i[:, :s + 1] = x_i[a, :s + 1]
            W = np.full(shape=N, fill_value=1.0 / N)
        
        x_i[:, s + 1] = x_i[:, s]

        for _ in range(MCMC_steps[s]):
            # MCMC step
            x_i[:, s + 1] = MCMC_update(
                x_i=x_i[:, s + 1], rng=rng, α=α_s[s + 1]
            )

    return x_i[rng.choice(a=N, p=W), :], log_Z, np.sum(W * h(x_i[:, -1]))


def conditional_SMC(
    x, N, sample_from_prior, log_likelihood, MCMC_update, setup, h=default_h,
    systematic_resampling=True, resample_threshold=0.5,
    rng=np.random.default_rng()
):
    """
    Single conditional SMC step with `N` particles

    `x` is the state on which is conditioned.
    See the function `SMC` for a description of the other parameters.
    """
    α_s, MCMC_steps = setup
    m = len(α_s) - 1

    # Initialize time s = 0.
    x_i = np.empty((N, m), dtype=object)
    x_i[0, :] = x

    if α_s[0] == 0.0:
        log_Z = 0.0

        for i in range(1, N):
            x_i[i, 0] = sample_from_prior(rng)
    else:
        x_i[1:, 0] = sample_from_prior(N - 1, α_s[0], rng)
        w_log = α_s[0] * log_likelihood(x_i[:, 0])
        w_log_max = w_log.max()
        w_log -= w_log_max
        W = np.exp(w_log)
        W_sum = W.sum() / N
        log_Z = np.log(W_sum) + w_log_max

    W = np.full(shape=N, fill_value=1.0 / N)

    for s in range(m):
        W, log_Z = get_weights(s, x_i, W, log_Z, α_s, log_likelihood)

        if s == m - 1:
            break
        
        if resample(W, resample_threshold):
            if systematic_resampling:
                a = cond_sys_resampling(W, rng)[1:]
            else:
                a = sample(prob=W, uniform_samples=rng.random(size=N - 1))
            
            x_i[1:, :s + 1] = x_i[a, :s + 1]
            W = np.full(shape=N, fill_value=1.0 / N)
        
        x_i[1:, s + 1] = x_i[1:, s]

        for _ in range(MCMC_steps[s]):
            # MCMC step
            x_i[1:, s + 1] = MCMC_update(
                x_i=x_i[1:, s + 1], rng=rng, α=α_s[s + 1]
            )

    return x_i[rng.choice(a=N, p=W), :], log_Z, np.sum(W * h(x_i[:, -1]))


def coupled_conditional_SMC(
    z, N, sample_from_prior, log_likelihood, coupled_MCMC_update, setup,
    h=default_h, systematic_resampling=True, resample_threshold=0.5,
    rng=np.random.default_rng()
):
    """
    Coupled conditional sequential Monte Carlo (SMC)

    One step each in two coupled Markov chains that follow
    from iterated conditional SMC with `N` particles.

    `z` contains both states on which is conditioned.

    `coupled_MCMC_update` is a Python function that takes a 2-dimensional NumPy
    array of objects `z_i` as input and outputs a NumPy array of objects of the
    same shape. Additionally, it has as parameters the `np.Generator` instance
    `rng` and the tempering temperature `α`.

    See the function `SMC` for a description of the other parameters.
    """
    α_s, MCMC_steps = setup
    m = len(α_s) - 1 # Number of "time" points of the SMC

    # Initialize time s = 0.
    z_i = np.empty((N, 2, m), dtype=object)
    z_i[0, :, :] = z
    
    if α_s[0] == 0.0:
        for i in range(1, N):
            z_i[i, 0, 0] = sample_from_prior(rng)
    else:
        z_i[1:, 0, 0] = sample_from_prior(N - 1, α_s[0], rng)

    z_i[1:, 1, 0] = z_i[1:, 0, 0]
    W_both = np.full(shape=(N, 2), fill_value=1.0 / N)

    if α_s[0] == 0.0:
        log_Z_both = np.zeros(2)
    else:
        log_Z_both = np.empty(2)

        for j in range(2):
            w_log = α_s[0] * log_likelihood(z_i[:, j, 0])
            w_log_max = w_log.max()
            w_log -= w_log_max
            W = np.exp(w_log)
            W_sum = W.sum() / N
            log_Z_both[j] = np.log(W_sum) + w_log_max

    for s in range(m):
        for j in range(2):
            W_both[:, j], log_Z_both[j] = get_weights(
                s, z_i[:, j, :], W_both[:, j], log_Z_both[j], α_s,
                log_likelihood
            )

        if s == m - 1:
            break
        
        resample_both = [resample(
            W_both[: j], resample_threshold
        ) for j in range(2)]
        
        if all(resample_both):
            if systematic_resampling:
                a = coupled_cond_sys_resampling(W_both, rng)[:, 1:]
            else:
                a = coupled_sampling(
                    p1=W_both[:, 0], p2=W_both[:, 1], n=N - 1, rng=rng
                )
            
            for j in range(2):
                z_i[1:, j, :s + 1] = z_i[a[j, :], j, :s + 1]
            
            W_both = np.full(shape=(N, 2), fill_value=1.0 / N)
        else:
            for j in range(2):
                if resample_both[j]:
                    if systematic_resampling:
                        a = cond_sys_resampling(W_both[:, j], rng)[1:]
                    else:
                        a = sample(
                            prob=W_both[:, j],
                            uniform_samples=rng.random(size=N - 1)
                        )

                    z_i[1:, j, :s + 1] = z_i[a, j, :s + 1]
                    W_both[:, j] = np.full(shape=N, fill_value=1.0 / N)
        
        z_i[1:, :, s + 1] = z_i[1:, :, s]

        for _ in range(MCMC_steps[s]):
            z_i[1:, :, s + 1] = coupled_MCMC_update(
                z_i=z_i[1:, :, s + 1], rng=rng, α=α_s[s + 1]
            )

    a = coupled_sampling(p1=W_both[:, 0], p2=W_both[:, 1], n=1, rng=rng)[:, 0]

    for j in range(2):
        z[j, :] = z_i[a[j], j, :]
    
    return z, log_Z_both, [
        np.sum(W_both[:, j] * h(z_i[:, j, -1])) for j in range(2)
    ]



def PIMH(
    x, log_Z, h_est, N, sample_from_prior, log_likelihood, MCMC_update, setup,
    h=default_h, systematic_resampling=True, resample_threshold=0.5,
    rng=np.random.default_rng()
):
    """
    One step in the Markov chain defined by particle independent Metropolis-
    Hastings

    `x` contains the current state.

    `log_Z` is the corresponding unbiased SMC estimate of the marginal
    likelihood.

    `h_est` is an estimate of the statistic `h` from the same SMC that produced
    `x`.

    See the function `SMC` for a description of the other parameters.
    """
    x_prop, log_Z_prop, h_prop = SMC(
        N, sample_from_prior, log_likelihood, MCMC_update, setup, h,
        systematic_resampling, resample_threshold, rng
    )
    
    if np.log(rng.random()) < log_Z_prop - log_Z:
        x, log_Z, h_est = x_prop, log_Z_prop, h_prop
    
    return x, log_Z, h_est


def coupled_PIMH(
    z, log_Z_both, h_both, N, sample_from_prior, log_likelihood, MCMC_update,
    setup, h=default_h, systematic_resampling=True, resample_threshold=0.5,
    rng=np.random.default_rng()
):
    """
    Coupled particle independent Metropolis-Hastings

    This is one MCMC step in Algorithm 3 from Middleton et al. (2019).
    `z` contains both current states.

    `log_Z_both` contains both corresponding unbiased SMC estimates of the
    marginal likelihood.

    `h_both` contains both estimates of the statistic `h` from the same SMCs
    that produced `z`.

    See the function `SMC` for a description of the other parameters.
    """    
    x_prop, log_Z_prop, h_prop = SMC(
        N, sample_from_prior, log_likelihood, MCMC_update, setup, h,
        systematic_resampling, resample_threshold, rng
    )

    mask = np.log(rng.random()) < log_Z_prop - log_Z_both
    log_Z_both[mask] = log_Z_prop
        
    for j in mask.nonzero()[0]:
        h_both[j] = h_prop
        z[j, :] = x_prop
    
    return z, log_Z_both, h_both


def coupled_MCMC(
    min_iter, max_iter, equal, N, sample_from_prior,
    log_likelihood, MCMC_update, setup, coupled_MCMC_update=None, h=default_h,
    systematic_resampling=True, resample_threshold=0.5,
    rng=np.random.default_rng(), PIMH_prob=1.0
):
    """
    Coupled MCMC via a mixture of conditonal SMC and PIMH

    This function returns two coupled Markov chains at leat up until their
    meeting time. They can be used for Rhee-Glynn estimation with respect to
    the posterior.

    `PIMH_prob` is the probability with which PIMH is used. For instance,
    `PIMH_prob = 0.0` results in using only conditional SMC.

    `min_iter` is the minimum number of steps that the MCMC chain is run for.
    The chain is run up to the meeting time or up to `min_iter`, whichever is
    greater.

    `max_iter` is the maximum number of steps that the MCMC chain is run for.
    This number is only reach if the coupled chains fail to meet.

    `equal` is a Python function that takes a 2-D NumPy array with two particle
    states as input and outputs as a Boolean whether they are equal.

    `coupled_MCMC_update` is a Python function that takes a 2-dimensional NumPy
    array of objects `z_i` as input and outputs a NumPy array of objects of the
    same shape. Additionally, it has as parameters the `np.Generator` instance
    `rng` and the tempering temperature `α`. The argument is only required if
    `PIMH_prob < 1.0`.

    See the function `SMC` for a description of the other parameters.

    This function output a tuple with the meeting time, the coupled estimates
    of the statistic `h`, the computation time used to get the MCMC chains to
    meet, and the computation time to run the MCMC until `min_iter` from the
    meeting time.
    """
    if PIMH_prob < 1.0 and coupled_MCMC_update is None:
        raise ValueError(
            "`coupled_MCMC_update` must be given if `PIMH_prob < 1.0`."
        )

    start_time = time.perf_counter()
    # Initialize by drawing from the SMC.
    z = np.empty((2, len(setup[0]) - 1), dtype=object)
    log_Z_both = np.empty(2)
    h_both = [None, None]
    
    z[1, :], log_Z_both[1], h_both[1] = SMC(
        N, sample_from_prior, log_likelihood, MCMC_update, setup, h,
        systematic_resampling, resample_threshold, rng
    )
    
    # Take one coupled MCMC step in the first chain.
    if rng.random() < PIMH_prob:
        z[0, :], log_Z_both[0], h_both[0] = SMC(
            N, sample_from_prior, log_likelihood, MCMC_update, setup, h,
            systematic_resampling, resample_threshold, rng
        )

        # The proposal is the initialization of the second chain.
        if np.log(rng.random()) < log_Z_both[1] - log_Z_both[0]:
            z[0, :] = z[1, :]
            log_Z_both[0] = log_Z_both[1]
            h_both[0] = h_both[1]
    else:
        z[0, :], log_Z_both[0], h_both[0] = conditional_SMC(
            z[1, :], N, sample_from_prior, log_likelihood, MCMC_update, setup,
            h, systematic_resampling, resample_threshold, rng
        )
    
    h_coupled = [h_both.copy()]
    
    for ii in range(max_iter):
        coupled = equal(z)
        
        if coupled:
            break
        
        if rng.random() < PIMH_prob:
            z, log_Z_both, h_both = coupled_PIMH(
                z, log_Z_both, h_both, N, sample_from_prior, log_likelihood,
                MCMC_update, setup, h, systematic_resampling,
                resample_threshold, rng
            )
        else:
            z, log_Z_both, h_both = coupled_conditional_SMC(
                z, N, sample_from_prior, log_likelihood, coupled_MCMC_update,
                setup, h, systematic_resampling, resample_threshold, rng
            )
        
        h_coupled.append(h_both.copy())
    
    meet_time = time.perf_counter()
    meeting_time = ii + 1
    
    if not coupled:
        warnings.warn("Failed to meet: The estimator is biased!")
    
    x, log_Z, h_est = z[0, :], log_Z_both[0], h_coupled[-1][0]

    for _ in range(meeting_time, min_iter):
        if rng.random() < PIMH_prob:
            x, log_Z, h_est = PIMH(
                x, log_Z, h_est, N, sample_from_prior, log_likelihood,
                MCMC_update, setup, h, systematic_resampling,
                resample_threshold, rng
            )
        else:
            x, log_Z, h_est = conditional_SMC(
                x, N, sample_from_prior, log_likelihood, MCMC_update, setup, h,
                systematic_resampling, resample_threshold, rng
            )

        h_coupled.append([h_est, None])
    
    end_time = time.perf_counter()

    return {
        "meeting time": meeting_time,
        "h coupled": np.array(h_coupled, dtype=float),
        "meet time": meet_time - start_time,
        "additional time": end_time - meet_time
    }


def _repeat_cMCMC(
    S, pyfunc, rng=np.random.default_rng()
):
    tmp_seed = random_seed(rng)

    def par_func(ss):
        return pyfunc(rng=np.random.default_rng(
            seed=np.random.SeedSequence(entropy=tmp_seed, spawn_key=(ss,))
        ))

    start_time = time.perf_counter()
    result_list = p_tqdm.p_umap(par_func, range(S))

    print("Elapsed time:", IPython.core.magics.execution._format_time(
        time.perf_counter() - start_time
    ))

    comp_time  = np.array([[
        result_list[s][key] for key in ["meet time", "additional time"]
    ] for s in range(S)])

    tau_arr = np.array([result_list[s]["meeting time"] for s in range(S)])
    rep_h_coupled = [result_list[s]["h coupled"] for s in range(S)]

    return {"meeting time": tau_arr, "h": rep_h_coupled, "time": comp_time}


def repeat_cMCMC(
    S, min_iter, max_iter, equal, N, sample_from_prior,
    log_likelihood, MCMC_update, setup, coupled_MCMC_update=None, h=default_h,
    systematic_resampling=True, resample_threshold=0.5,
    rng=np.random.default_rng(), PIMH_prob=1.0
):
    """
    Repeat `coupled_MCMC` `S` times in parallel for unbiased estimation

    See the function `coupled_MCMC` for a description of the other parameters.
    """
    def pyfunc(rng):
        return coupled_MCMC(
            min_iter, max_iter, equal, N, sample_from_prior, log_likelihood,
            MCMC_update, setup, coupled_MCMC_update, h, systematic_resampling,
            resample_threshold, rng, PIMH_prob
        )

    return _repeat_cMCMC(S, pyfunc, rng)


# We use a function from the R package `LaplacesDemon`.
if not rpackages.isinstalled("LaplacesDemon"):
    rpackages.importr("utils").install_packages(
        "LaplacesDemon", repos="https://cloud.r-project.org"
    )

LaplacesDemon = rpackages.importr("LaplacesDemon")
rpy2.robjects.numpy2ri.activate() # Enable passing NumPy arrays to R.


def get_IAT(result, burnin):
    """
    Compute the integrated autocorrelation time (IAT).

    `result` is the output from the function `repeat_cMCMC`.

    `burnin` is the number of iterations that is discarded when computing the
    IAT.
    """
    S = len(result["meeting time"])
    IAT = np.empty(S)
    
    for s in range(S):
        IAT[s] = LaplacesDemon.IAT(np.array([
            h_coupled[0] for h_coupled in result["h"][s][burnin:]
        ]))[0]
    
    return IAT