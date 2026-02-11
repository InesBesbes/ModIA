linKern <- function(x, y, param){
  # input:
  #  x,y: input vectors
  #  param: parameters (sigma)
  # output:
  #  kern: covariance matrix cov(x,y)
  sigma <- param[1]
  kern <- sigma^2*outer(x, y, '*')
  return(kern)
}

cosKern <- function(x, y, param){
  # input:
  #  x,y: input vectors
  #  param: parameters (sigma,theta)
  # output:
  #  kern: covariance matrix cov(x,y)
  sigma <- param[1]
  theta <- param[2]
  dist <- outer(x/theta, y/theta, '-')
  kern <- sigma^2*cos(dist)
  return(kern)
}

expKern <- function(x, y, param){
  # input:
  #  x,y: input vectors
  #  param: parameters (sigma,theta)
  # output:
  #  kern: covariance matrix cov(x,y)
  sigma <- param[1]
  theta <- param[2]
  dist <- outer(x/theta, y/theta, '-')
  kern <- sigma^2*exp(-abs(dist))
  return(kern)
}

mat5_2Kern <- function(x,y,param){
  sigma <- param[1]
  theta <- param[2]
  dist <- outer(x/theta, y/theta, '-')
  kern <- sigma^2 * (1+sqrt(5)*abs(dist) + 5/3 * abs(dist)^2)*exp(-sqrt(5)*abs(dist))
  return(kern)
}

mat3_2Kern <- function(x,y,param){
  sigma <- param[1]
  theta <- param[2]
  dist <- outer(x/theta, y/theta, '-')
  kern <- sigma^2 * (1+sqrt(3)*abs(dist))*exp(-sqrt(3)*abs(dist))
  return(kern)
}

squared_expKern <- function(x,y,param){
  sigma <- param[1]
  theta <- param[2]
  dist <- outer(x/theta, y/theta, '-')
  kern <- sigma^2 *exp(-1/2 * dist^2)
  return(kern)
}

