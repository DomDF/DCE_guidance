functions {
    
  real gumbel_copula_lpdf(row_vector uv, real theta) {
    
    real neg_log_u = -log(uv[1]);
    real log_neg_log_u = log(neg_log_u);
    real neg_log_v = -log(uv[2]);
    real log_neg_log_v = log(neg_log_v);
    real log_temp = log_sum_exp(theta * log_neg_log_u, theta * log_neg_log_v);
    real theta_m1 = theta - 1;
  
    if (theta < 1) 
      reject("theta must be >= 1");
    
    if (is_inf(theta)) {
      if (uv[1] == uv[2]) 
        return 0;
      else 
        return negative_infinity();
    }
  
    return theta_m1 * log_neg_log_u + theta_m1 * log_neg_log_v + neg_log_u + neg_log_v
           - exp(log_temp / theta)
           + log_sum_exp(2 * theta_m1 / -theta * log_temp,
                         log(theta_m1) + (1 - 2 * theta) / theta * log_temp);
    }
}

data {
  
  int <lower = 0> n_data;
  vector <lower = 0> [n_data] yield;
  vector <lower = 0> [n_data] tensile;
  
  real prior_logmean_yield_mean;
  real <lower = 0> prior_logmean_yield_sd;
  real <lower = 0> prior_logsd_yield_rate;
  
  real prior_logmean_tensile_mean;
  real <lower = 0> prior_logmean_tensile_sd;
  real <lower = 0> prior_logsd_tensile_rate;
  
  real <lower = 0> prior_theta_rate;

}

parameters {

  real logmean_yield;
  real <lower = 0> logsd_yield;
  real logmean_tensile;
  real <lower = 0> logsd_tensile;
  real <lower = 0> gumbel_theta;

}

// transformed parameters {
//   
//   real <lower = 0, upper = 1> yt_ratio = yield / tensile;
//   
// }

model {

  // 
  target += lognormal_lpdf(yield | logmean_yield, logsd_yield);
  target += lognormal_lpdf(tensile | logmean_tensile, logsd_tensile);
  
  for (n in 1:n_data){
    target += gumbel_copula_lpdf([lognormal_cdf(yield[n] | logmean_yield, logsd_yield),
                                  lognormal_cdf(tensile[n] | logmean_tensile, logsd_tensile)] | 
                                  gumbel_theta);
  }
                             
  // Priors
  target += normal_lpdf(logmean_yield | prior_logmean_yield_mean, prior_logmean_yield_sd);
  target += exponential_lpdf(logsd_yield | prior_logsd_yield_rate);
  
  target += normal_lpdf(logmean_tensile | prior_logmean_tensile_mean, prior_logmean_tensile_sd);
  target += exponential_lpdf(logsd_tensile | prior_logsd_tensile_rate);
  
  target += exponential_lpdf(gumbel_theta | prior_theta_rate);
}

generated quantities{
  
  vector [n_data] log_lik;
  
  for (n in 1:n_data) {
    log_lik[n] = gumbel_copula_lpdf([lognormal_cdf(yield[n] | logmean_yield, logsd_yield),
                                     lognormal_cdf(tensile[n] | logmean_tensile, logsd_tensile)] |
                                     gumbel_theta);
  }
  
}

