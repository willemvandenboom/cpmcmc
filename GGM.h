#include <array>

#include <igraph/igraph.h>

void rgwish_L_cpp(
    double* K_out, igraph_t* G_ptr, double df, double* rate_in, long seed
);

void rgwish_L_identity_cpp(
    double* K_out, igraph_t* G_ptr, double df, long seed
);

std::array<std::array<int, 2>, 2> sample_e_both_cpp(
    igraph_t* G0_ptr, igraph_t* G1_ptr, long seed
);

bool update_G_cpp(
    double* K_out, igraph_t* G_ptr, bool add, std::vector<int> e,
    double edge_prob, double df, double df_0, double* rate_in, long seed
);

std::tuple<bool, bool> update_G_both_cpp(
    double* K0_out, double* K1_out, igraph_t* G0_ptr, igraph_t* G1_ptr,
    bool add0, bool add1, std::vector<int> e0, std::vector<int> e1,
    double edge_prob, double df, double df_0, double* rate_in, long seed
);

void rejection_sampling_cpp(
    double* K_out, double* adj_out, int p, int n, int N, double alpha,
    double edge_prob, double df_0, double* rate_in, long seed
);