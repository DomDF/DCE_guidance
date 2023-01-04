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
  
  real <lower = 0> strength [n_strength];
  real <lower = 0> strength_m;
  real <lower = 0> strength_sd;

}

transformed parameters{
  
  // vector <lower = 0> [n_strength] strength = strength_meas + error;
  
}

model {
  
  // Gaussian model for material strength
  target += normal_lpdf(strength | strength_m, strength_sd);

  // Measurement error
  target += normal_lpdf(strength_meas | strength, error);
  
  
  // Priors
  target += normal_lpdf(strength_m | m_s, sd_s);
  target += exponential_lpdf(strength_sd | rate_s);
  
}

generated quantities{

  real strength_post_pred = normal_rng(strength_m, strength_sd);

}
