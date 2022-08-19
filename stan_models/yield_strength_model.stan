functions {
  
}

data {

  int <lower = 1> n_strength; // number of strength measurements
  vector <lower = 0> [n_strength] strength_meas; // strength measurements
  
  // Measurement error
  real <lower = 0> error;
  
  // Prior parameters
  real m_s;
  real <lower = 0> sd_s;
  real <lower = 0> rate_s;

}

parameters {
  
  real strength;
  real strength_m_log;
  real <lower = 0> strength_sd_log;

}

model {
  
  // Measurement error
  target += normal_lpdf(strength_meas | strength, error);
  
  // Log normal model for material strength
  target += lognormal_lpdf(strength | strength_m_log, strength_sd_log);
  
  // Priors
  target += normal_lpdf(strength_m_log | m_s, sd_s);
  target += exponential_lpdf(strength_sd_log | rate_s);
  
}

generated quantities{

  real strength_post_pred = lognormal_rng(strength_m_log, strength_sd_log);
  
}
