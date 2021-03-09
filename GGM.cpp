#include <array>
#include <cmath>
#include <numeric>
#include <tuple>
#include <vector>

// The distributions in `<random>` are not portable. That is, they do not
// yield the same random numbers on different machines. Therefore, we use the
// distributions from Boost, which are protable.
// This code was tested using Boost version 1.74.0 (Boost.org).
#include <boost/random/chi_squared_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/uniform_int_distribution.hpp>

// This code was tested using Blaze version 3.8.0
// (https://bitbucket.org/blaze-lib/blaze/src/master/) with the fix from this
// pull request: https://bitbucket.org/blaze-lib/blaze/pull-requests/46.
#include <blaze/math/DynamicMatrix.h>
#include <blaze/math/LowerMatrix.h>
#include <blaze/math/UpperMatrix.h>
#include <blaze/math/DynamicVector.h>
#include <blaze/math/Column.h>
#include <blaze/math/Columns.h>
#include <blaze/math/Row.h>
#include <blaze/math/Rows.h>
#include <blaze/math/Submatrix.h>

// This code was tested using igraph version 0.8.5 (https://igraph.org).
#include <igraph/igraph.h>

/*
C++'s RNGs are not very fast. Therefore, I use the RNG from
https://gist.github.com/martinus/c43d99ad0008e11fcdbf06982e25f464:
extremely fast random number generator that also produces very high quality
random. see PractRand: http://pracrand.sourceforge.net/PractRand.txt
*/
class sfc64 {
  public:
    using result_type = uint64_t;

    static constexpr uint64_t(min)() { return 0; }
    static constexpr uint64_t(max)() { return UINT64_C(-1); }

    sfc64() : sfc64(std::random_device{}()) {}

    explicit sfc64(uint64_t seed) : m_a(seed), m_b(seed), m_c(seed), m_counter(1) {
        for (int i = 0; i < 12; ++i) {
            operator()();
        }
    }

    uint64_t operator()() noexcept {
        auto const tmp = m_a + m_b + m_counter++;
        m_a = m_b ^ (m_b >> right_shift);
        m_b = m_c + (m_c << left_shift);
        m_c = rotl(m_c, rotation) + tmp;
        return tmp;
    }

  private:
    template <typename T> T rotl(T const x, int k) { return (x << k) | (x >> (8 * sizeof(T) - k)); }

    static constexpr int rotation = 24;
    static constexpr int right_shift = 11;
    static constexpr int left_shift = 3;
    uint64_t m_a;
    uint64_t m_b;
    uint64_t m_c;
    uint64_t m_counter;
};


std::array<int, 2> sample_edge(igraph_t* G_ptr, sfc64& rng) {
    boost::random::uniform_int_distribution<int>
        r_edge(0, igraph_ecount(G_ptr) - 1);

    int from, to;
    igraph_edge(G_ptr, r_edge(rng), &from, &to);
    return {from, to};
}


std::array<std::array<int, 2>, 2> sample_e_both_cpp(
    igraph_t* G0_ptr, igraph_t* G1_ptr, long seed
) {
    // Coupled sampling of edges
    igraph_t G_intersect;
    igraph_intersection(&G_intersect, G0_ptr, G1_ptr, nullptr, nullptr);
    double n_intersect = igraph_ecount(&G_intersect);
    std::array<std::array<int, 2>, 2> e_both;
    sfc64 rng(seed);

    if (n_intersect == 0.0) {
        igraph_destroy(&G_intersect);
        e_both[0] = sample_edge(G0_ptr, rng);
        e_both[1] = sample_edge(G1_ptr, rng);
        return e_both;
    }

    /*
    Decide whether the same or different edges are sampled.
    We use Algorithm 1 from Jasra et al. (2017, doi:10.1137/17M1111553):
    We maximize the probability that the edges are the same while the edges
    still marginally are sampled uniformly at random from each graph.
    */
    std::array<double, 2> p = {
        n_intersect / igraph_ecount(G0_ptr),
        n_intersect / igraph_ecount(G1_ptr),
    };

    bool G1_larger = p[0] < p[1];
    double alpha = p[not G1_larger] + 1.0 - p[G1_larger];
    boost::random::uniform_01<double> runif;

    if (runif(rng) < alpha) {
        if (runif(rng) < p[not G1_larger] / alpha) {
            // The edges are the same.
            e_both.fill(sample_edge(&G_intersect, rng));
        } else for (int i = 0; i < 2; i++) {
            igraph_t G_min;

            if (i) {
                igraph_difference(&G_min, G1_ptr, &G_intersect);
            } else {
                igraph_difference(&G_min, G0_ptr, &G_intersect);
            }

            e_both[i] = sample_edge(&G_min, rng);
            igraph_destroy(&G_min);
        }
    } else {
        e_both[G1_larger] = sample_edge(&G_intersect, rng);
        igraph_t G_min;

        if (G1_larger) {
            igraph_difference(&G_min, G0_ptr, &G_intersect);
        } else {
            igraph_difference(&G_min, G1_ptr, &G_intersect);
        }
        
        e_both[not G1_larger] = sample_edge(&G_min, rng);
        igraph_destroy(&G_min);
    }
    
    igraph_destroy(&G_intersect);
    return e_both;
}


blaze::DynamicMatrix<double> inv_pos_def(blaze::DynamicMatrix<double> mat) {
    blaze::invert<blaze::byLLH>(mat);
    return mat;
}


template <class T>
auto submatrix_view(
    blaze::DynamicMatrix<double>& mat, std::vector<T>& ind_row,
    std::vector<T>& ind_col
) {
    return columns(rows(mat, ind_row), ind_col);
}


template <class T>
auto submatrix_view_square(
    blaze::DynamicMatrix<double>& mat, std::vector<T>& ind
) {
    return submatrix_view(mat, ind, ind);
}


bool is_complete(igraph_t* G_ptr) {
    int p = igraph_vcount(G_ptr);
    return igraph_ecount(G_ptr) == p * (p - 1) / 2;
};


blaze::UpperMatrix<blaze::DynamicMatrix<double> > rwish_identity_chol(
    int p, double df, sfc64& rng
) {
    blaze::UpperMatrix<blaze::DynamicMatrix<double> > Phi(p);
    boost::random::normal_distribution<> rnorm(0.0, 1.0);
    df += p - 1;
    
    // Generate the upper-triangular Cholesky decompositon of a standard
    // Wishart random variable.
    for (int i = 0; i < p; i++) {
        boost::random::chi_squared_distribution<> rchisq(df - i);
        Phi(i, i) = std::sqrt(rchisq(rng));
        for (int j = i + 1; j < p; j++) Phi(i, j) = rnorm(rng);
    }
    
    return Phi;
}


blaze::DynamicMatrix<double> rwish_identity(int p, double df, sfc64& rng) {
    /*
    Sample a `p` by `p` matrix from a Wishart distribution with `df` degrees of
    freedom with an identity rate matrix.
    */

    blaze::UpperMatrix<blaze::DynamicMatrix<double> >
        Phi = rwish_identity_chol(p, df, rng);
    
    return declsym(trans(Phi) * Phi);
}


blaze::DynamicMatrix<double> rwish(
    double df, blaze::DynamicMatrix<double>& rate, sfc64& rng
) {
    blaze::UpperMatrix<blaze::DynamicMatrix<double> >
        Phi_identity = rwish_identity_chol(rate.rows(), df, rng);

    blaze::LowerMatrix<blaze::DynamicMatrix<double> > chol;
    llh(rate, chol);
    blaze::invert(chol);
    blaze::DynamicMatrix<double> Phi = Phi_identity * chol;
    return declsym(trans(Phi) * Phi);
}


blaze::DynamicMatrix<double> rginvwish_L_body(
    igraph_t* G_ptr, blaze::DynamicMatrix<double> W
) {
    // This function follows Section 2.4 in Lenkoski (2013, arXiv:1304.1350v1).
    blaze::DynamicMatrix<double> Sigma = inv_pos_def(W);
    if (is_complete(G_ptr)) return Sigma;
    int p = W.rows();
    W = Sigma;  // Step 1

    // Inspired by C++ code from the R package BDgraph
    blaze::DynamicMatrix<double> W_previous(p, p);
    std::vector<std::vector<double> > neighbors(p);
    // Avoid recomputing the neighbors for each iteration:
    igraph_vector_ptr_t igraph_neighbors;
    igraph_vector_ptr_init(&igraph_neighbors, p);

    igraph_neighborhood(
        G_ptr, &igraph_neighbors, igraph_vss_all(), 1, IGRAPH_ALL, 1
    );

    for (int i = 0; i < p; i++) {
        igraph_vector_t* N_ptr
            = (igraph_vector_t*) igraph_vector_ptr_e(&igraph_neighbors, i);

        neighbors[i].resize(igraph_vector_size(N_ptr));

        for (int j = 0; j < neighbors[i].size(); j++)
            neighbors[i][j] = igraph_vector_e(N_ptr, j);
    }

    IGRAPH_VECTOR_PTR_SET_ITEM_DESTRUCTOR(
        &igraph_neighbors, igraph_vector_destroy
    );

    igraph_vector_ptr_destroy_all(&igraph_neighbors);
    
    for (int i = 0; i < 10000; i++) {
        W_previous = W;
        
        for (int j = 0; j < p; j++) if (neighbors[j].size() > 0) {
            blaze::DynamicVector<double> beta_star = blaze::solve(
                blaze::declsym(submatrix_view_square(W, neighbors[j])),
                column(rows(Sigma, neighbors[j]), j)
            );

            for (int k = 0; k < p; k++) {
                if (k == j) continue;
                double tmp_sum = 0.0;

                for (int l = 0; l < neighbors[j].size(); l++)
                    tmp_sum += W(k, neighbors[j][l]) * beta_star[l];

                W(j, k) = tmp_sum;
                W(k, j) = tmp_sum;
            }
        } else for (int k = 0; k < p; k++) {
            if (k == j) continue;
            W(j, k) = 0.0;
            W(k, j) = 0.0;
        }

        // 1e-8 is consistent with BDgraph.
        if (blaze::mean(blaze::abs(W - W_previous)) < 1e-8) return W;
    }

    return W;
}


blaze::DynamicMatrix<double> rginvwish_L(
	igraph_t* G_ptr, double df, blaze::DynamicMatrix<double>& rate, sfc64& rng
) {
	/*
    Sample from the G-inverse-Wishart using the Lenkoski method.
    
    `rate` is the inverse of the scale matrix of the Wishart distribution.
    This function follows Section 2.4 in Lenkoski (2013, arXiv:1304.1350v1).
    */
    return rginvwish_L_body(G_ptr, rwish(df, rate, rng));
}


blaze::DynamicMatrix<double> rgwish_L(
    igraph_t* G_ptr, double df, blaze::DynamicMatrix<double>& rate, sfc64& rng
) {
    /*
    Sample from the G-Wishart using the Lenkoski method.
    
    `rate` is the inverse of the scale matrix of the Wishart distribution.
    This function follows Section 2.4 in Lenkoski (2013, arXiv:1304.1350v1).
    */
    if (is_complete(G_ptr)) return rwish(df, rate, rng);
    return inv_pos_def(rginvwish_L(G_ptr, df, rate, rng));
}


blaze::DynamicMatrix<double> rgwish_L(
    igraph_t* G_ptr, blaze::DynamicMatrix<double>& rate,
    blaze::DynamicMatrix<double>& wish
) {
    /*
    Sample from the G-Wishart using the Lenkoski method.
    
    `rate` is the inverse of the scale matrix of the Wishart distribution.
    This function follows Section 2.4 in Lenkoski (2013, arXiv:1304.1350v1).
    */
    return inv_pos_def(rginvwish_L_body(G_ptr, wish));
}


blaze::DynamicMatrix<double> rginvwish_L_identity(
    igraph_t* G_ptr, double df, sfc64& rng
) {
    return rginvwish_L_body(
        G_ptr, rwish_identity(igraph_vcount(G_ptr), df, rng)
    );
}


blaze::DynamicMatrix<double> rgwish_L_identity(
    igraph_t* G_ptr, double df, sfc64& rng
) {
    if (is_complete(G_ptr))
        return rwish_identity(igraph_vcount(G_ptr), df, rng);

    return inv_pos_def(rginvwish_L_identity(G_ptr, df, rng));
}


blaze::DynamicMatrix<double> rgwish_L_identity(
    igraph_t* G_ptr, blaze::DynamicMatrix<double>& wish
) {
    return inv_pos_def(rginvwish_L_body(G_ptr, wish));
}


void rgwish_L_identity_cpp(
    double* K_out, igraph_t* G_ptr, double df, long seed
) {
    sfc64 rng(seed);
    int p = igraph_vcount(G_ptr);
    blaze::DynamicMatrix<double> K = rgwish_L_identity(G_ptr, df, rng);

    // Copy `K` to `K_out`.
    for (int i = 0; i < p; i++) {
        int ixp = i * p;
        for (int j = 0; j < p; j++) K_out[ixp + j] = K(i, j);
    }
}


void rgwish_L_cpp(
    double* K_out, igraph_t* G_ptr, double df, double* rate_in, long seed
) {
    sfc64 rng(seed);
    int p = igraph_vcount(G_ptr);
    blaze::DynamicMatrix<double> rate(p, p, rate_in);
    blaze::DynamicMatrix<double> K = rgwish_L(G_ptr, df, rate, rng);

    // Copy `K` to `K_out`.
    for (int i = 0; i < p; i++) {
        int ixp = i * p;
        for (int j = 0; j < p; j++) K_out[ixp + j] = K(i, j);
    }
}


double proposal_G_es(int p, int n_e_tilde, int n_e) {
    // Proposal transition probability from `G` to `G_tilde` based on edge
    // counts

    if (n_e == 0 or n_e == p * (p - 1) / 2)
        return 2.0 / p / (p - 1);

    if (n_e > n_e_tilde)
        return 0.5 / n_e;

    return 0.5 / (p*(p - 1)/2 - n_e);
}



auto update_G_subbody(
    igraph_t* G_ptr, bool add, std::vector<int> e, double edge_prob,
    double* rate_in,
    blaze::UpperMatrix<blaze::DynamicMatrix<double> >& wish_identity_chol,
    blaze::DynamicMatrix<double>& wish_0, double Z, double log_unif
) {
    int p = igraph_vcount(G_ptr);

    // Reorder vertices such that Φ and Φ_tilde only differ in element
    // (p - 1, p).
    std::vector<int> perm(p), perm_inv(p);
    std::iota(perm.begin(), perm.end(), 0);

    // Permute the nodes involved in `e`.
    if (e[0] != p - 2) {
        perm[e[0]] = p - 2;

        if (e[1] == p - 2) {
            perm[p - 2] = p - 1;
            perm[p - 1] = e[0];
        } else {
            perm[p - 2] = e[0];
            perm[p - 1] = e[1];
            perm[e[1]] = p - 1;
        }
    }

    for (int i = 0; i < p; i++) perm_inv[perm[i]] = i;

    blaze::DynamicMatrix<double> rate(p, p, rate_in),
        rate_perm = submatrix_view_square(rate, perm_inv);

    igraph_t G_perm;
    igraph_vector_t igraph_vec;
    std::vector<double> perm_d(p);
    for (int i = 0; i < p; i++) perm_d[i] = perm[i];

    igraph_permute_vertices(
        G_ptr, &G_perm, igraph_vector_view(&igraph_vec, perm_d.data(), p)
    );

    // Sample auxiliary variables.
    blaze::LowerMatrix<blaze::DynamicMatrix<double> > Phi, Phi_0, U;
    llh(rate_perm, U);
    blaze::LowerMatrix<blaze::DynamicMatrix<double> > U_inv = U;
    blaze::invert(U_inv);
    blaze::DynamicMatrix<double> Phi_tmp = wish_identity_chol * U_inv;
    blaze::DynamicMatrix<double> wish = declsym(trans(Phi_tmp) * Phi_tmp);
    blaze::DynamicMatrix<double> K = rgwish_L(&G_perm, rate_perm, wish);

    if (add) {
        igraph_add_edge(&G_perm, p - 2, p - 1);
    } else {
        igraph_es_t es;
        igraph_es_pairs_small(&es, false, p - 2, p - 1, -1);
        igraph_delete_edges(&G_perm, es);
        igraph_es_destroy(&es);
    }

    blaze::DynamicMatrix<double>
        K_0_tilde = rgwish_L_identity(&G_perm, wish_0);

    igraph_destroy(&G_perm);
    llh(K, Phi);
    llh(K_0_tilde, Phi_0);
    boost::random::normal_distribution<> rnorm(0.0, 1.0);

    double gamma = Phi(p - 1, p - 2), theta_tilde = Phi_0(p - 1, p - 2),
        mean = -Phi(p - 2, p - 2) / rate_perm(p - 1, p - 1) * rate_perm(p - 1, p - 2),
        var = 1.0 / rate_perm(p - 1, p - 1),
        tmp_sum = 0.0;  // Log of acceptance rate

    if (add) {
        tmp_sum += 0.5 * (std::log(var) + Z*Z - theta_tilde*theta_tilde);
        Phi(p - 1, p - 2) = mean + std::sqrt(var)*Z;

        Phi_0(p - 1, p - 2) = -sum(submatrix(
            Phi_0, p - 2, 0, 1, p - 2
        ) % submatrix(Phi_0, p - 1, 0, 1, p - 2)) / Phi_0(p - 2, p - 2);
    } else {
        tmp_sum += 0.5 * (Z*Z - std::log(var) - std::pow(gamma - mean, 2)/var);
        Phi_0(p - 1, p - 2) = Z;

        Phi(p - 1, p - 2) = -sum(submatrix(
            Phi, p - 2, 0, 1, p - 2
        ) % submatrix(Phi, p - 1, 0, 1, p - 2)) / Phi(p - 2, p - 2);
    }

    int tmp = 2*add - 1, n_e = igraph_ecount(G_ptr);

    if (edge_prob <= 0.0) {
        // Size-based prior
        int r = p * (p - 1) / 2;

        if (add) {
            tmp_sum += std::log(n_e + 1) - std::log(r - n_e);
        } else {
            tmp_sum += std::log(r - n_e + 1) - std::log(n_e);
        }

        // Negative `edge_prob` specifies a truncated geometric prior with
        // success probability -`edge_prob` on the number of edges.
        if (edge_prob < 0.0) tmp_sum += tmp * std::log(1 + edge_prob);
    } else if (edge_prob != 0.5) {
        // From the prior on G
        tmp_sum += tmp * (std::log(edge_prob) - std::log(1.0 - edge_prob));
    }

    // Transition probabilities
    tmp_sum += std::log(proposal_G_es(p, n_e, n_e + tmp))
        - std::log(proposal_G_es(p, n_e + tmp, n_e));

    tmp_sum += tmp * (
        std::log(Phi(p - 2, p - 2)) - std::log(Phi_0(p - 2, p - 2))
    );


    // What follows equals either one of the following 2 lines:
    // tmp_sum -= 0.5 * trace_inner(K_tilde - K, rate_perm)
    // tmp_sum -= 0.5 * np.sum((U @ Φ_tilde[-2, :])**2 - (U @ Φ[-2, :])**2)
    double delta = Phi(p - 1, p - 2) - gamma;

    tmp_sum -= delta*Phi(p - 2, p - 2)*sum(
        submatrix(U, p - 1, 0, 1, p - 1) % submatrix(U, p - 2, 0, 1, p - 1)
    ) + (gamma + 0.5*delta)*delta*sqrNorm(row(U, p - 1));

    tmp_sum -= 0.5 * (
        std::pow(Phi_0(p - 1, p - 2), 2) - std::pow(theta_tilde, 2)
    );

    return std::make_tuple(log_unif < tmp_sum, K, Phi, perm, p);
}


auto update_G_body(
    igraph_t* G_ptr, bool add, std::vector<int> e,
    double edge_prob, double df, double df_0, double* rate_in, long seed
) {
    sfc64 rng(seed);
    int p = igraph_vcount(G_ptr);

    blaze::UpperMatrix<blaze::DynamicMatrix<double> >
        wish_identity_chol = rwish_identity_chol(p, df, rng);

    blaze::DynamicMatrix<double> wish_0 = rwish_identity(p, df_0, rng);
    boost::random::normal_distribution<> rnorm(0.0, 1.0);
    boost::random::uniform_01<double> runif;
    double Z = rnorm(rng), log_unif = std::log(runif(rng));

    return update_G_subbody(
        G_ptr, add, e, edge_prob, rate_in, wish_identity_chol, wish_0, Z,
        log_unif
    );
}


bool update_G_cpp(
    double* K_out, igraph_t* G_ptr, bool add, std::vector<int> e,
    double edge_prob, double df, double df_0, double* rate_in, long seed
) {
    bool res;
    blaze::DynamicMatrix<double> K;
    blaze::LowerMatrix<blaze::DynamicMatrix<double> > Phi;
    std::vector<int> perm;
    int p;
    
    std::tie(res, K, Phi, perm, p)
        = update_G_body(G_ptr, add, e, edge_prob, df, df_0, rate_in, seed);

    if (res) K = declsym(Phi * trans(Phi));
    K = submatrix_view_square(K, perm);

    // Copy `K` to `K_out`.
    for (int i = 0; i < p; i++) {
        int ixp = i * p;
        for (int j = 0; j < p; j++) K_out[ixp + j] = K(i, j);
    }

    return res;
}


std::tuple<bool, bool> update_G_both_cpp(
    double* K0_out, double* K1_out, igraph_t* G0_ptr, igraph_t* G1_ptr,
    bool add0, bool add1, std::vector<int> e0, std::vector<int> e1,
    double edge_prob, double df, double df_0, double* rate_in, long seed
) {
    sfc64 rng(seed);
    int p = igraph_vcount(G0_ptr);
    bool res0, res1;
    blaze::DynamicMatrix<double> K;
    blaze::LowerMatrix<blaze::DynamicMatrix<double> > Phi;
    std::vector<int> perm;

    blaze::UpperMatrix<blaze::DynamicMatrix<double> >
        wish_identity_chol = rwish_identity_chol(p, df, rng);

    blaze::DynamicMatrix<double> wish_0 = rwish_identity(p, df_0, rng);
    boost::random::normal_distribution<> rnorm(0.0, 1.0);
    boost::random::uniform_01<double> runif;
    double Z = rnorm(rng), log_unif = std::log(runif(rng));

    std::tie(res0, K, Phi, perm, p) = update_G_subbody(
        G0_ptr, add0, e0, edge_prob, rate_in, wish_identity_chol, wish_0, Z,
        log_unif
    );

    if (res0) K = declsym(Phi * trans(Phi));
    K = submatrix_view_square(K, perm);

    // Copy `K` to `K0_out`.
    for (int i = 0; i < p; i++) {
        int ixp = i * p;
        for (int j = 0; j < p; j++) K0_out[ixp + j] = K(i, j);
    }

    std::tie(res1, K, Phi, perm, p) = update_G_subbody(
        G1_ptr, add1, e1, edge_prob, rate_in, wish_identity_chol, wish_0, Z,
        log_unif
    );

    if (res1) K = declsym(Phi * trans(Phi));
    K = submatrix_view_square(K, perm);

    // Copy `K` to `K1_out`.
    for (int i = 0; i < p; i++) {
        int ixp = i * p;
        for (int j = 0; j < p; j++) K1_out[ixp + j] = K(i, j);
    }

    return std::make_tuple(res0, res1);
}


void rejection_sampling_cpp(
    double* K_out, double* adj_out, int p, int n, int N, double alpha,
    double edge_prob, double df_0, double* rate_in, long seed
) {
    sfc64 rng(seed);
    boost::random::uniform_01<double> runif;
    int max_n_e = p * (p - 1) / 2;
    boost::random::uniform_int_distribution<int> runif_int(0, max_n_e);
    blaze::DynamicMatrix<double> rate(p, p, rate_in);
    igraph_rng_t igraph_rng;
    igraph_rng_init(&igraph_rng, &igraph_rngtype_mt19937);
    igraph_rng_seed(&igraph_rng, seed);
    igraph_rng_set_default(&igraph_rng);
    igraph_erdos_renyi_t type;
    igraph_real_t p_or_m;

    double tmp_prob_pow, log_tmp_prob, log_likelihood_max
        = 0.5 * n * (p*std::log(n) - std::log(det(rate)) - p);

    if (edge_prob <= 0.0) {
        // Use size-based prior
        type = IGRAPH_ERDOS_RENYI_GNM;

        if (edge_prob < 0.0) {
            // Negative `edge_prob` specifies a truncated geometric prior with
            // success probability -`edge_prob` on the number of edges.
            tmp_prob_pow = 1.0 - std::pow(1.0 + edge_prob, max_n_e);
            log_tmp_prob = std::log(1.0 + edge_prob);
        }
    } else {
        // Use independent edge prior
        type = IGRAPH_ERDOS_RENYI_GNP;
        p_or_m = edge_prob;
    }

    for (int s = 0; s < N; s++) {
        blaze::DynamicMatrix<double> K;
        igraph_t G;

        while (true) {
            if (edge_prob == 0.0) p_or_m = runif_int(rng);

            // Negative `edge_prob` specifies a truncated geometric prior with
            // success probability -`edge_prob` on the number of edges.
            if (edge_prob < 0.0)  p_or_m = std::floor(
                std::log(1.0 - runif(rng)*tmp_prob_pow) / log_tmp_prob
            );

            igraph_erdos_renyi_game(&G, type, p, p_or_m, false, false);
            K = rgwish_L_identity(&G, df_0, rng);

            if (std::log(runif(rng)) < alpha * (0.5*(
                n*std::log(det(rate)) - trace(K * rate)
            ) - log_likelihood_max)) break;

            igraph_destroy(&G);
        }

        igraph_matrix_t adj;
        igraph_matrix_init(&adj, 0, 0);
        igraph_get_adjacency(&G, &adj, IGRAPH_GET_ADJACENCY_UPPER, false);
        igraph_destroy(&G);
        int sxpxp = s * p * p;

        for (int i = 0; i < p; i++) {
            int sxpxp_ixp = sxpxp + i*p;

            for (int j = 0; j < p; j++) {
                int ind = sxpxp_ixp + j;
                K_out[ind] = K(i, j);
                adj_out[ind] = MATRIX(adj, i, j);
            }
        }

        igraph_matrix_destroy(&adj);
    }
}


int main() {
    return 0;
}