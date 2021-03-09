import ctypes
import os
import platform
import random
import subprocess
import time
import warnings

# See
# https://stackoverflow.com/questions/58909525/what-is-numpy-core-multiarray-umath-implement-array-function-and-why-it-costs-l
# 10% speedup by disabling additional NumPy array functionality:
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"
os.environ["EXTRA_CLING_ARGS"] = "-I/usr/local/include"

# To prevent spawning of more than 1 thread: Actually faster in this case!
os.environ["MKL_NUM_THREADS"] = "1"

import cppyy
import igraph
import IPython.core.magics.execution as IPy_exec
import numpy as np
import p_tqdm
import sklearn.datasets

import cpmcmc


warnings.filterwarnings("ignore", category=DeprecationWarning) 
platform_name = platform.system()

if platform_name == "Darwin":
    compiler_opts = "g++ -march=native"
elif platform_name == "Linux":
    # "native" yields "architecture not recognized" on the Yale-NUS server.
    compiler_opts = "/usr/bin/clang++-10 -march=nocona"
else:
    raise NotImplementedError(
        "Compiling has only been tested on macOS and Linux."
    )

binary_file_name = "GGM_" + platform_name + ".so"

# cppyy seems to be using the wrong BLAS on the Yale-NUS server. We therefore
# explicitly specify the path to Intel MKL's LAPACK and BLAS taken from
# `np.show_config()`.
subprocess.run(
    compiler_opts \
        + " -shared -fPIC -I/usr/local/include -std=c++17 " \
        + "-L$CONDA_PREFIX/lib -L/usr/local -Wl,-rpath,$CONDA_PREFIX/lib " \
        + "-lmkl_rt -ligraph -O2 -DNDEBUG -o " + binary_file_name \
        + " GGM.cpp",
    shell=True, check=True
)

cppyy.include("GGM.h")
cppyy.load_library(binary_file_name)

# Reduce Python overhead:
rgwish_L_identity_cpp = cppyy.gbl.rgwish_L_identity_cpp
update_G_cpp = cppyy.gbl.update_G_cpp
update_G_both_cpp = cppyy.gbl.update_G_both_cpp
rejection_sampling_cpp = cppyy.gbl.rejection_sampling_cpp
sample_e_both_cpp = cppyy.gbl.sample_e_both_cpp
c_einsum = np.core._multiarray_umath.c_einsum

# As the metabolite data is not public, we use another example data set here.
iris = sklearn.datasets.load_iris()

data = iris["data"][(
    iris["target"] == (iris["target_names"] == "virginica").nonzero()[0][0]
).nonzero()[0], :]

data -= data.mean(axis=0)
n, p = data.shape
U = data.T @ data
# The code uses the size-based prior if edge_prob <= 0.0. Negative `edge_prob`
# specifies a truncated geometric prior with success probability `-edge_prob`
# on the number of edges. `edge_prob = 0` specifies a uniform prior on the
# number of edges. If `edge_prob > 0.0`, then the edges are a priori
# independent with prior edge inclusion probability `edge_prob`.
edge_prob = 0.5
df_0 = 3.0  # Degrees of freedom of the Wishart prior
rate_0 = np.eye(p)  # Rate matrix of the Wishart prior
rng = np.random.Generator(np.random.SFC64(seed=0))


N = 10**2  # Number of SMC particles
k = 7
min_iter = 40  # Denoted by l in the paper
max_iter = 10**4
tmp_seed = cpmcmc.random_seed(rng)
n_hours = 0.02  # Time budget post adaptation in hours


def trace_inner(A, B):
    """
    Trace inner product of the matrices `A` and `B`

    This function computes <A, B> = tr(A' B).
    Taken from https://stackoverflow.com/a/18855277/5216563.
    `A` can be a higher dimensional array. Then, this function is broadcast
    over the first dimensions of `A`.
    """
    return c_einsum('...ij,ji->...', A, B)


def rgwish_identity(G, df, rng):
    """Sample from the G-Wishart distribution with an identity scale matrix."""
    K = np.empty(2 * (G.vcount(),))
    
    try:
        rgwish_L_identity_cpp(
            K, G.__graph_as_capsule(), df, cpmcmc.random_seed(rng)
        )
    except:
        print("Error in `rgwish_L_identity_cpp`. Retrying...")
        return rgwish_identity(G, df, rng)
    
    return K


def sample_from_prior(rng):
    """
    Draw z = (G, K) from its prior.
    
    This sets the random number generator from iGraph.
    """
    igraph.set_random_number_generator(random.Random(cpmcmc.random_seed(rng)))
    
    if edge_prob <= 0.0:
        m_max_p1 = p*(p - 1)//2 + 1
        
        if edge_prob == 0.0:
            # Use the size-based prior.
            m = rng.integers(m_max_p1)
        else:
            # Use the size-based prior with a truncated geometric distribution
            # on the graph size.
            prob = -edge_prob
            
            m = int(np.floor(np.log(
                1.0 - rng.random()*(1.0 - (1.0 - prob)**m_max_p1)
            ) / np.log(1.0 - prob)))
        
        G = igraph.Graph.Erdos_Renyi(n=p, m=m)
    else:
        G = igraph.Graph.Erdos_Renyi(n=p, p=edge_prob)
    
    return [G, rgwish_identity(G, df_0, rng)]


def log_likelihood(x_i):
    """
    Log-likelihood of the Gaussian graphical model
    
    This function returns a NumPy array of the log of the likelihood where
    constant terms are dropped.
    """
    N = len(x_i)
    K_i = np.concatenate([x_i[i][1] for i in range(N)]).reshape(N, p, p)
    return 0.5 * (n*np.linalg.slogdet(K_i)[1] - trace_inner(K_i, U))


def sample_edge(G, rng):
    """Sample an edge from graph `G`."""
    return G.es[rng.choice(G.ecount())].tuple


def update_G(G, add, e, α, rng):
    """
    MCMC step for the given graph `G`

    `G_tilde` is the proposed graph.

    This follows the exchange algorithm  from
    Lenkoski (2013, arXiv:1304.1350v1).
    The proposal distribution is assumed to be adding/removing an edge with
    equal probability, if `G` is not empty nor full.
    Then, the edge `e` is selected uniformly at random.
    """
    df = df_0 + α*n
    rate = rate_0 + α*U
    K = np.empty((p, p))
    
    try:
        accept = update_G_cpp(
            K, G.__graph_as_capsule(), add, e, edge_prob, df, df_0, rate,
            cpmcmc.random_seed(rng)
        )
    except:
        print("Error in `update_G_cpp`. Retrying...")
        return update_G(G, add, e, α, rng)
    
    if accept:
        G_tilde = G.copy()
        G_tilde[e] = True if add else False
        return [G_tilde, K]
    else:
        return [G, K]


def MCMC_update(x_i, rng, α=1.0):
    if type(x_i[0]) is not igraph.Graph:
        for i in range(len(x_i)):
            x_i[i] = MCMC_update(x_i[i], rng, α)
        
        return x_i

    # MCMC step
    # Decide whether to propose an edge addition or removal.
    if x_i[0].ecount() == 0:
        add = True
    elif x_i[0].ecount() == p * (p - 1) // 2:
        add = False
    else:
        add = rng.random() < 0.5

    # Algorithm from Section 3.2 of arXiv:1304.1350v1
    # Metropolis-Hastings step to add edge to `G`
    # Pick edge to add or remove.
    if add:
        e = sample_edge(~x_i[0], rng=rng)
    else:
        e = sample_edge(x_i[0], rng=rng)

    return update_G(G=x_i[0], add=add, e=e, α=α, rng=rng)


def G_equal(z):
    """Check whether both graphs in `z` are equal"""
    G_both = [z[0][0], z[1][0]]

    if G_both[0].ecount() != G_both[1].ecount():
        return False
    
    es_both = [G_both[j].es for j in range(2)]
    
    for i in range(G_both[0].ecount()):
        if es_both[0][i] != es_both[1][i]:
            return False
    
    return True


def equal(z):
    """Check whether both particles in `z` are equal."""
    K_both = [np.concatenate([x[1] for x in z[j, :]]) for j in range(2)]
    coupled = np.allclose(K_both[0], K_both[1])
        
    if np.allclose(K_both[0], K_both[1]):
        for s in range(z.shape[1]):
            if not G_equal(z[:, s]):
                return False

        return True
    
    return False

    if z.ndim == 2:
        for s in range(z.shape[1]):
            if not equal(z[:, s]):
                return False

        return True

    G_both = [z[0][0], z[1][0]]

    if G_both[0].ecount() != G_both[1].ecount():
        return False
    
    es_0 = G_both[0].es
    es_1 = G_both[1].es
    
    for i in range(G_both[0].ecount()):
        if es_0[i] != es_1[i]:
            return False
    
    return True


def sample_e_both(G_both, rng):
    """Coupled sampling of edges"""
    res = sample_e_both_cpp(
        G_both[0].__graph_as_capsule(),
        G_both[1].__graph_as_capsule(), cpmcmc.random_seed(rng)
    )
    
    return [tuple(res[i]) for i in range(2)]


def update_G_both(G_both, add_both, e_both, α, rng):
    df = df_0 + α*n
    rate = rate_0 + α*U
    K0 = np.empty((p, p))
    K1 = np.empty((p, p))
    
    try:
        accept_both = update_G_both_cpp(
            K0, K1, G_both[0].__graph_as_capsule(),
            G_both[1].__graph_as_capsule(), add_both[0], add_both[1],
            e_both[0], e_both[1], edge_prob, df, df_0, rate,
            cpmcmc.random_seed(rng)
        )
    except:
        print("Error in `update_G_both_cpp`. Retrying...")
        return update_G_both(G_both, add_both, e_both, α, rng)
    
    K_both = (K0, K1)
    res = np.empty(2, dtype=object)
    
    for j in range(2):    
        if accept_both[j]:
            G_tilde = G_both[j].copy()
            G_tilde[e_both[j]] = True if add_both[j] else False
            res[j] = [G_tilde, K_both[j]]
        else:
            res[j] = [G_both[j], K_both[j]]
    
    return res


def coupled_MCMC_update(z_i, rng, α=1.0):
    if z_i.ndim == 2:
        for i in range(len(z_i)):
            z_i[i, :] = coupled_MCMC_update(z_i[i, :], rng, α)

        return z_i

    # Coupled MCMC step                    
    # If both graphs are equal, ensure that the MCMC step doesn't change that.
    if G_equal(z_i):
        z_i[0] = MCMC_update(z_i[0], rng, α)
        z_i[1] = z_i[0]
        return z_i
    
    # Decide whether to propose an edge addition or removal.
    # For maximal coupling, we would like the same edge to be
    # added/removed in both graphs.
    G_both = [z_i[j][0] for j in range (2)]
    add_both = np.empty(2, dtype=bool)
    tmp_add = rng.random() < 0.5
    
    for j in range(2):
        if G_both[j].ecount() == 0:
            add_both[j] = True
        elif G_both[j].ecount() == p * (p - 1) // 2:
            add_both[j] = False
        else:
            add_both[j] = tmp_add

    # Algorithm from Section 3.2 of arXiv:1304.1350v1
    # Metropolis-Hastings step to add edge to `G`
    if add_both.all():
        # Pick edge to add.
        e_both = sample_e_both([~G_both[j] for j in range(2)], rng)
    elif not add_both.any():
        e_both = sample_e_both(G_both, rng)  # Pick edge to remove.
    else:
        e_both = [sample_edge(
            ~G_both[j] if add_both[j] else G_both[j], rng
        ) for j in range(2)]
        
    return update_G_both(G_both, add_both, e_both, α, rng)


def rejection_sampling(N, α, rng):
    K_out = np.empty((N, p, p))
    adj_out = np.empty((N, p, p))
    
    try:
        rejection_sampling_cpp(
            K_out, adj_out, p, n, N, α, edge_prob, df_0, U,
            cpmcmc.random_seed(rng)
        )
    except:
        print("Error in `rejection_sampling`. Retrying...")
        return rejection_sampling(N, α, rng)
    
    res = np.empty(N, dtype=object)
    
    for i in range(N):
        res[i] = [
            igraph.Graph.Adjacency(adj_out[i, :, :].tolist(), mode=1),
            K_out[i, :, :]
        ]
    
    return res


def test_func(x_i):
    N = len(x_i)
    ret = np.empty(N, dtype=int)
    
    for i in range(N):
        ret[i] = x_i[i][0].ecount()
    
    return ret


setup = cpmcmc.adapt_SMC(
    10**3, sample_from_prior, log_likelihood, MCMC_update, test_func, rng=rng,
    log_likelihood_max=0.5 * n * (p*np.log(n) - np.linalg.slogdet(U)[1] - p)
)


def GGM_SMC(
    N, sample_from_prior, log_likelihood, MCMC_update, setup,
    systematic_resampling=True, resample_threshold=0.5,
    rng=np.random.default_rng()
):
    """
    Run SMC with `N` particles modified for Gaussian graphical models to output
    the posterior edge inclusion probabilities.

    See `cpmcmc.SMC` for a description of parameters.
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
        W, log_Z = cpmcmc.get_weights(s, x_i, W, log_Z, α_s, log_likelihood)

        if s == m - 1:
            break
        
        if cpmcmc.resample(W, resample_threshold):
            if systematic_resampling:
                a = cpmcmc.sys_resampling(w=W, U=rng.random())
            else:
                a = cpmcmc.sample(prob=W, uniform_samples=rng.random(size=N))
            
            x_i[:, :s + 1] = x_i[a, :s + 1]
            W = np.full(shape=N, fill_value=1.0 / N)
        
        x_i[:, s + 1] = x_i[:, s]

        for _ in range(MCMC_steps[s]):
            # MCMC step
            x_i[:, s + 1] = MCMC_update(
                x_i=x_i[:, s + 1], rng=rng, α=α_s[s + 1]
            )

    post_edge_prob = np.zeros((p, p))
    
    for i in range(N):
        post_edge_prob += W[i] * np.array(x_i[i, -1][0].get_adjacency().data)

    return x_i[rng.choice(a=N, p=W), :], log_Z, post_edge_prob



def GGM_PIMH(
    x, log_Z, post_edge_prob, N, sample_from_prior, log_likelihood,
    MCMC_update, setup, systematic_resampling=True, resample_threshold=0.5,
    rng=np.random.default_rng()
):
    """
    One step in the Markov chain defined by particle independent Metropolis-
    Hastings modified for Gaussian graphical models to output the posterior
    edge inclusion probabilities.

    See `cpmcmc.PIMH` for a description of the parameters.
    """
    x_prop, log_Z_prop, post_edge_prob_prop = GGM_SMC(
        N, sample_from_prior, log_likelihood, MCMC_update, setup,
        systematic_resampling, resample_threshold, rng
    )
    
    if np.log(rng.random()) < log_Z_prop - log_Z:
        x, log_Z, post_edge_prob = x_prop, log_Z_prop, post_edge_prob_prop
    
    return x, log_Z, post_edge_prob


def GGM_coupled_PIMH(
    z, log_Z_both, edge_prob_both, N, sample_from_prior, log_likelihood,
    MCMC_update, setup, systematic_resampling=True, resample_threshold=0.5,
    rng=np.random.default_rng()
):
    """
    Coupled particle independent Metropolis-Hastings modified for Gaussian
    graphical models to output the posterior edge inclusion probabilities.

    See `cpmcmc.PIMH` for a description of the parameters.
    """    
    x_prop, log_Z_prop, post_edge_prob_prop = GGM_SMC(
        N, sample_from_prior, log_likelihood, MCMC_update, setup,
        systematic_resampling, resample_threshold, rng
    )

    mask = np.log(rng.random()) < log_Z_prop - log_Z_both
    log_Z_both[mask] = log_Z_prop
        
    for j in mask.nonzero()[0]:
        edge_prob_both[j] = post_edge_prob_prop
        z[j, :] = x_prop
    
    return z, log_Z_both, edge_prob_both


def GGM_conditional_SMC(
    x, N, sample_from_prior, log_likelihood, MCMC_update,
    setup, systematic_resampling=True, resample_threshold=0.5,
    rng=np.random.default_rng()
):
    """
    Single conditional SMC step with `N` particles modified for Gaussian
    graphical models to output the posterior edge inclusion probabilities.

    `x` is the state on which is conditioned.
    See the function `cpmcmc.SMC` for a description of the other parameters.
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
        W, log_Z = cpmcmc.get_weights(s, x_i, W, log_Z, α_s, log_likelihood)

        if s == m - 1:
            break
        
        if cpmcmc.resample(W, resample_threshold):
            if systematic_resampling:
                a = cpmcmc.cond_sys_resampling(W, rng)[1:]
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

    post_edge_prob = np.zeros((p, p))
    
    for i in range(N):
        post_edge_prob += W[i] * np.array(x_i[i, -1][0].get_adjacency().data)

    return x_i[rng.choice(a=N, p=W), :], log_Z, post_edge_prob


def GGM_coupled_conditional_SMC(
    z, N, sample_from_prior, log_likelihood, coupled_MCMC_update, setup,
    systematic_resampling=True, resample_threshold=0.5,
    rng=np.random.default_rng()
):
    """
    Coupled conditional sequential Monte Carlo (SMC) modified for Gaussian
    graphical models to output the posterior edge inclusion probabilities.

    See `cpmcmc.coupled_conditional_SMC` for a description of the parameters.
    """
    α_s, MCMC_steps = setup
    m = len(α_s) - 1 # Number of "time" points of the SMC

    # Initialize time s = 0.
    z_i = np.empty((N, 2, m), dtype=object)
    z_i[0, :, :] = z
    
    if α_s[0] == 0.0:
        log_Z_both[0] = 0.0

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
            W_both[:, j], log_Z_both[j] = cpmcmc.get_weights(
                s, z_i[:, j, :], W_both[:, j], log_Z_both[j], α_s,
                log_likelihood
            )

        if s == m - 1:
            break
        
        resample_both = [cpmcmc.resample(
            W_both[: j], resample_threshold
        ) for j in range(2)]
        
        if all(resample_both):
            if systematic_resampling:
                a = cpmcmc.coupled_cond_sys_resampling(W_both, rng)[:, 1:]
            else:
                a = cpmcmc.coupled_sampling(
                    p1=W_both[:, 0], p2=W_both[:, 1], n=N - 1, rng=rng
                )
            
            for j in range(2):
                z_i[1:, j, :s + 1] = z_i[a[j, :], j, :s + 1]
            
            W_both = np.full(shape=(N, 2), fill_value=1.0 / N)
        else:
            for j in range(2):
                if resample_both[j]:
                    if systematic_resampling:
                        a = cpmcmc.cond_sys_resampling(W_both[:, j], rng)[1:]
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

    a = cpmcmc.coupled_sampling(
        p1=W_both[:, 0], p2=W_both[:, 1], n=1, rng=rng
    )[:, 0]

    for j in range(2):
        z[j, :] = z_i[a[j], j, :]

    edge_prob_both = [np.zeros((p, p)) for j in range(2)]
    
    for j in range(2):
        for i in range(N):
            edge_prob_both[j] += W_both[i, j] * np.array(
                z_i[i, j, -1][0].get_adjacency().data
            )
    
    return z, log_Z_both, edge_prob_both


def GGM_coupled_MCMC(
    min_iter, max_iter, equal, N, sample_from_prior,
    log_likelihood, MCMC_update, setup, coupled_MCMC_update=None,
    systematic_resampling=True, resample_threshold=0.5,
    rng=np.random.default_rng(), PIMH_prob=1.0, time_limit=np.inf
):
    """
    Coupled MCMC via a mixture of conditonal SMC and PIMH modified for Gaussian
    graphical models to output the posterior edge inclusion probabilities.
    
    `time_limit` is the time in seconds after which this function is
    terminated.

    See `cpmcmc.coupled_MCMC` for a description of the other parameters.
    """
    if PIMH_prob < 1.0 and coupled_MCMC_update is None:
        raise ValueError(
            "`coupled_MCMC_update` must be given if `PIMH_prob < 1.0`."
        )

    start_time = time.perf_counter()
    # Initialize by drawing from the SMC.
    z = np.empty((2, len(setup[0]) - 1), dtype=object)
    log_Z_both = np.empty(2)
    edge_prob_both = [None, None]
    
    z[1, :], log_Z_both[1], edge_prob_both[1] = GGM_SMC(
        N, sample_from_prior, log_likelihood, MCMC_update, setup,
        systematic_resampling, resample_threshold, rng
    )

    if time.perf_counter() > time_limit:
        print("Hit time limit during SMC initialization.")
        return
    
    # Take one coupled MCMC step in the first chain.
    if rng.random() < PIMH_prob:
        z[0, :], log_Z_both[0], edge_prob_both[0] = GGM_SMC(
            N, sample_from_prior, log_likelihood, MCMC_update, setup,
            systematic_resampling, resample_threshold, rng
        )

        # The proposal is the initialization of the second chain.
        if np.log(rng.random()) < log_Z_both[1] - log_Z_both[0]:
            z[0, :] = z[1, :]
            log_Z_both[0] = log_Z_both[1]
            edge_prob_both[0] = edge_prob_both[1]
    else:

        z[0, :], log_Z_both[0], edge_prob_both[0] = GGM_conditional_SMC(
            z[1, :], N, sample_from_prior, log_likelihood, MCMC_update, setup,
            systematic_resampling, resample_threshold, rng
        )

    if time.perf_counter() > time_limit:
        print("Hit time limit during first outer MCMC step.")
        return

    edge_prob_coupled = [edge_prob_both.copy()]
    
    for ii in range(max_iter):
        coupled = equal(z)
        
        if coupled:
            if ii > 2:
                print("Coupled after", ii, "steps.")

            break

        if time.perf_counter() > time_limit:
            print("Hit time limit at step", ii)
            return
        
        if ii % 50 == 0 and ii > 0:
            print("Not yet met after", ii, "steps.")
        
        if rng.random() < PIMH_prob:
            z, log_Z_both, edge_prob_both = GGM_coupled_PIMH(
                z, log_Z_both, edge_prob_both, N, sample_from_prior,
                log_likelihood, MCMC_update, setup, systematic_resampling,
                resample_threshold, rng
            )
        else:
            z, log_Z_both, edge_prob_both = GGM_coupled_conditional_SMC(
                z, N, sample_from_prior, log_likelihood, coupled_MCMC_update,
                setup, systematic_resampling, resample_threshold, rng
            )
        
        edge_prob_coupled.append(edge_prob_both.copy())
    
    meet_time = time.perf_counter()
    meeting_time = ii + 1
    
    if not coupled:
        warnings.warn("Failed to meet: The estimator is biased!")
    
    x, log_Z = z[0, :], log_Z_both[0]
    post_edge_prob = edge_prob_coupled[-1][0]

    for ii in range(meeting_time, min_iter):
        if rng.random() < PIMH_prob:
            x, log_Z, post_edge_prob = GGM_PIMH(
                x, log_Z, post_edge_prob, N, sample_from_prior, log_likelihood,
                MCMC_update, setup, systematic_resampling, resample_threshold,
                rng
            )
        else:
            x, log_Z, post_edge_prob = GGM_conditional_SMC(
                x, N, sample_from_prior, log_likelihood, MCMC_update, setup,
                systematic_resampling, resample_threshold, rng
            )

        edge_prob_coupled.append([post_edge_prob, None])

        if time.perf_counter() > time_limit:
            print(
                "Hit time limit at step", ii,
                "which is beyond the meeting time."
            )

            return
    
    end_time = time.perf_counter()

    return edge_prob_coupled, meeting_time
    # return {
    #     "meeting time": meeting_time,
    #     "h coupled": np.array(h_coupled, dtype=float),
    #     "meet time": meet_time - start_time,
    #     "additional time": end_time - meet_time
    # }


def post_edge_prob_fun(s):
    rng=np.random.default_rng(
        seed=np.random.SeedSequence(entropy=tmp_seed, spawn_key=(s,))
    )

    post_edge_prob_list = []
    meeting_time_list = []
    result_list = []
    time_limit = time.perf_counter() + n_hours*60.0**2

    while time.perf_counter() < time_limit:
        result = GGM_coupled_MCMC(
            min_iter, max_iter, equal, N, sample_from_prior=rejection_sampling,
            log_likelihood=log_likelihood, MCMC_update=MCMC_update,
            setup=setup, coupled_MCMC_update=coupled_MCMC_update, rng=rng,
            PIMH_prob=1.0, time_limit=time_limit
        )
        
        if result is None:  # The time budget is up.
            break
        
        result_list.append(result)
        meeting_time = result[1]      
        post_edge_prob = np.zeros((p, p))

        # Ergodic average
        for i in range(k - 1, min_iter):
            post_edge_prob += result[0][i][0]

        post_edge_prob /= min_iter - k + 1

        # Bias correction
        tmp = np.zeros((p, p))

        for i in range(k, meeting_time):
            tmp += min(min_iter - k + 1, i + 1 - k) * (
                result[0][i][0] - result[0][i][1]
            )

        post_edge_prob_list.append(post_edge_prob + tmp/(min_iter - k + 1))
        meeting_time_list.append(meeting_time)
    
    return post_edge_prob_list, meeting_time_list, result_list


S = 8  # The number of CPU cores used
tmp_seed = cpmcmc.random_seed(rng)
start_time = time.perf_counter()

print(
    "Starting computation with", n_hours, "hour time limit at",
    time.asctime(time.gmtime()), "UTC"
)

result_list = p_tqdm.p_umap(post_edge_prob_fun, range(S))
print("Elapsed time:", IPy_exec._format_time(time.perf_counter() - start_time))


meeting_time = []

for s in range(S):
    for i in range(len(result_list[s][0])):
        meeting_time.append(result_list[s][1][i])

if len(meeting_time) > 0:
    print("Median:",  np.median(meeting_time))
    print("Max:",  np.max(meeting_time))
    post_edge_prob = np.empty((len(meeting_time), p, p))
    ind = 0

    for s in range(S):
        for i in range(len(result_list[s][0])):
            post_edge_prob[ind, :, :] = result_list[s][0][i]
            ind += 1

    print("Posterior edge inclusion probabilities:")
    print(post_edge_prob.mean(axis=0))
    print("Monte Carlo standard error:")
    print(post_edge_prob.std(axis=0, ddof=1) / np.sqrt(S))