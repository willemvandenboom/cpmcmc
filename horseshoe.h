#include <array>
#include <cmath>
#include <numeric>

// The distributions in `<random>` are not portable. That is, they do not
// yield the same random numbers on different machines. Therefore, we use the
// distributions from Boost, which are protable.
// This code was tested using Boost version 1.74.0 (Boost.org).
#include <boost/random/uniform_01.hpp>


double f(double x, double eps) {
	return eps*x + std::log1p(x);
}


double rexp_trunc(double lam, double lower, double upper, double u) {
    // Sampling from an exponential distribution truncated to [`lower`,`upper`]
    return lower - std::log1p(-u + u*std::exp(-lam *(upper - lower)))/lam;
}


void sample_h_eps(double* z_out, double* eps_vec, int p, long seed) {
	/*
	Sample from the density proportional to exp(-eps * t) / (1 + t), t > 0.

	We do this for j = 0,...,p-1 with eps=`eps_vec[j]`.
	It uses the rejection sampler in Appendix S1 of
	http://jmlr.org/papers/v21/19-536.html.
	*/

	std::mt19937_64 rng(seed);
	boost::random::uniform_01<double> runif;
	double a = 0.2, b = 10.0;

	for (int j = 0; j < p; j++) {
		double z, f_L, eps = eps_vec[j], A = f(a / eps, eps),
			I = f(1.0 / eps, eps), B = f(b / eps, eps),
			lam_2 = (I - A) / (1.0 - a) * eps,
			lam_3 = (B - I) / (b - 1.0) * eps;

		std::array<double, 4> nu = {
			std::log1p(a / eps), std::exp(-A) / lam_2 * -std::expm1(A - I),
			std::exp(-I) / lam_3 * -std::expm1(I - B), std::exp(-B) / eps
		};

		std::partial_sum(nu.begin(), nu.end(), nu.begin());

		do {
			double tmp = nu[3] * runif(rng);

			if (tmp < nu[0]) {
	            z = std::pow(1.0 + a/eps, runif(rng)) - 1.0;
	            f_L = std::log1p(z);
	        } else if (tmp < nu[1]) {
	            z = rexp_trunc(lam_2, a / eps, 1.0 / eps, runif(rng));
	            f_L = A + lam_2*(z - a/eps);
	        } else if (tmp < nu[2]) {
	            z = rexp_trunc(lam_3, 1.0 / eps, b / eps, runif(rng));
	            // The next `1.0` is a `b` in the paper, but I believe that
	            // `1.0` is correct.
	            f_L = I + lam_3*(z - 1.0/eps);
	        } else {
	            z = rexp_trunc(eps, b / eps, INFINITY, runif(rng));
	            f_L = B + eps*(z - b/eps);
	        }
		} while (std::log(runif(rng)) > f_L - f(z, eps));
		
		z_out[j] = z;
	}
}


double sample_P(double m, double U, std::mt19937_64& rng) {
	boost::random::uniform_01<double> runif;
	double T = 1.0/U - 1.0;
	return -std::log1p(std::expm1(-m * T) * runif(rng)) / m;
}


double dens_P(double eta, double m, double U) {
	double T = 1.0/U - 1.0;
	if (eta > T) return 0.0;
	return m * std::exp(-m * eta) / -std::expm1(-m * T);
}


void coupled_sample_eta_cpp(
	double* eta0, double* eta1, double* m0, double* m1, int p, long seed
) {
	/*
	Coupled slice sampling for eta per Algorithm 4 of arXiv:2012.04798v1 with
	𝜈 = 1
	*/

	std::mt19937_64 rng(seed);
	boost::random::uniform_01<double> runif;

	for (int j = 0; j < p; j++) {
		double U_crn = runif(rng);
		double U0 = U_crn / (1.0 + eta0[j]);
		double U1 = U_crn / (1.0 + eta1[j]);
		eta0[j] = sample_P(m0[j], U0, rng);
		double W = runif(rng);

		if (dens_P(eta0[j], m0[j], U0) * W <= dens_P(eta0[j], m1[j], U1)) {
			eta1[j] = eta0[j];
			continue;
		}

		do {
			eta1[j] = sample_P(m1[j], U1, rng);
		} while (
			dens_P(eta1[j], m1[j], U1) * runif(rng)
				<= dens_P(eta1[j], m0[j], U0)
		);
	}
}