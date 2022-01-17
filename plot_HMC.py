import matplotlib.pyplot as plt
import numpy as np
import rpy2.robjects as robjects

import cpmcmc

robjects.r["load"]("HMC_comparison.RData")
result_r = robjects.r["results"]

S = len(result_r)
max_iter = len(result_r[0][1])
min_iter = max_iter
result_list_tmp = S * [None]
print("Putting the R list in the Python format needed...")

for s in range(S):
    print("Finished " + str(s) + " out of " + str(S), end="\r")
    tmp = result_r[s][0][0]
    meeting_time = int(tmp) if tmp != np.inf else max_iter - 1
    h_coupled = np.full((max_iter, 2), np.nan)
    
    for i in range(max_iter):
        for j in range(2):
            try:
                h_coupled[i, j] = result_r[s][1][i][j][0]
            except:
                pass
    
    meet_time = result_r[s][4][0]
    additional_time = result_r[s][5][0]
    
    result_list_tmp[s] = {
        "meeting time": meeting_time,
        "h coupled": np.array(h_coupled, dtype=float),
        "meet time": meet_time,
        "additional time": additional_time
    }



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


rng = np.random.default_rng(seed=0)

comp_time  = np.array([[
    result_list_tmp[s][key] for key in ["meet time", "additional time"]
] for s in range(S)])

tau_arr = np.array([result_list_tmp[s]["meeting time"] for s in range(S)])
rep_h_coupled = [result_list_tmp[s]["h coupled"] for s in range(S)]

result = {"meeting time": tau_arr, "h": rep_h_coupled, "time": comp_time}


result_list = [result]

n_bootstrap = 10**3
alpha_level = 0.3
col = "black"
n_setup = len(result_list)
h_est_all = n_setup * [None]


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

fig.savefig("HMC_comparison.pdf")