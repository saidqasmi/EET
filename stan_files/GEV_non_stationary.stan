functions {

  real gev_lpdf(vector y, vector mu, vector sigma, real xi) {
    real m;
    m = num_elements(y);
    real ll;
    if (xi != 0)
      ll = -sum(log(sigma)) - (1+1/xi)*sum(log(1+xi*(y-mu)./sigma)) - sum((1+xi*(y-mu)./sigma)^(-1/xi));
    else
      ll = -sum(log(sigma)) - sum((y-mu)./sigma) - sum(exp(-((y-mu)./sigma)));
    return ll;

  }
}

data {
  int N; 
  vector[N] Y; 
  int N_X; 
  vector[N_X] X; 
  vector[5] p_m ; 
  matrix[5,5] p_cov ; 
}


parameters {
  vector[5] tirage;
  
  


}
transformed parameters {
  vector[N] mu;
  vector<lower=0>[N] sigma;
  real<lower=-1, upper=1> xi;
  
  vector[5] para;
  para = p_m+ p_cov*tirage;
  mu =para[2]*X+para[1];
  sigma =exp(para[4]*X+para[3]);
  xi= para[5];
  
  


}


model {

    

    
tirage ~ normal(0, 1);
    

Y ~ gev(mu, sigma, xi);

}
