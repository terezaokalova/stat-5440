data {
  int<lower=1> N;  int<lower=1> D;  int<lower=2> K;
  array[N] vector[D] y;
}
parameters {
  vector<lower=0, upper=1>[K - 1] v;
  array[K] vector[D] mu;
  array[K] vector<lower=0>[D] sigma;
  real<lower=0> alpha;
}
transformed parameters {
  simplex[K] w;
  w[1] = v[1];
  for (k in 2:(K - 1))
    w[k] = v[k] * prod(1 - v[1:(k - 1)]);
  w[K] = 1 - sum(w[1:(K - 1)]);
}
model {
  alpha ~ gamma(2, 2);
  v ~ beta(1, alpha);
  for (k in 1:K) { mu[k] ~ normal(0, 3); sigma[k] ~ cauchy(0, 2); }
  for (n in 1:N) {
    vector[K] lps;
    for (k in 1:K) {
      lps[k] = log(w[k]);
      for (d in 1:D) lps[k] += normal_lpdf(y[n,d] | mu[k,d], sigma[k,d]);
    }
    target += log_sum_exp(lps);
  }
}
generated quantities {
  int n_active = 0;
  for (k in 1:K) if (w[k] > 0.05) n_active += 1;
}
