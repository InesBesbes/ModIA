rm(list=ls()) # to clear the environment

#### loading some packages and functions ####
library("plot3D")
library("MASS")
source("kernFun.R")
library(matlib)


#### Example with the Exp. kernel  ####
x <- seq(0, 1, 0.01) # regular grid
param <- c(1, 0.5) # covariance parameters
k1 <- expKern(x, x, param) # computing the covariance matrix using an exp. kernel
image2D(k1, theta = 0, xlab = "x", ylab = "y") # plotting the covariance matrix
# Q: what can you observe from the covariance matrix?
?mvrnorm # using the help from RStudio

k2 = mat5_2Kern(x,x,param)
k3 = mat3_2Kern(x,x,param)
k4 = squared_expKern(x,x,param)

## simulating some samples using the "mvrnorm" function
samples1 <- mvrnorm(n=1, mu = rep(0,101), Sigma = k1)
samples2 <- mvrnorm(n=1, mu = rep(0,101), Sigma = k2)
samples3 <- mvrnorm(n=1, mu = rep(0,101), Sigma = k3)
samples4 <- mvrnorm(n=1, mu = rep(0,101), Sigma = k4)

par(mfrow=c(2,2))

matplot(x=x, y=samples1, type="l", lty=1, main="Simulation de 100 échantillons (exponential kernel)")
matplot(x=x, y=samples3, type="l", lty=1, main="Simulation de 100 échantillons (matern 3/2)")
matplot(x=x, y=samples2, type="l", lty=1, main="Simulation de 100 échantillons (matern 5/2)")
matplot(x=x, y=samples4, type="l", lty=1, main="Simulation de 100 échantillons (squared exponential)")
par(mfrow=c(1,1))

# Q: what can you observe from the samples?

# theta = plus grand plus variables corrélées sur de longues distances 
# cst de lipschitz décroit si theta croit 
# nu : plus ça croit plus c'est lisse 
#rajouter polynome pour rajouter régularité  (matern)
#régularité liée à la régularité du kernel en 0

X = seq(0, 1, 1/6)
x_prime = seq(0, 1, 1/100)


f <- function(x){
  return(x+sin(4*pi*x))
}

Y = f(X)

condMean <- function(x, X, Y, kern, param) {
  return(kern(x, X, param)%*%inv(kern(X,X,param))%*%Y)
}

condCov <- function(x, x_prime, X,  kern, param) {
  return(kern(x, x_prime, param)-kern(x,X,param)%*%inv(kern(X,X,param))%*%kern(X,x_prime, param)) 
}

plot(X, Y,  col="blue", 
     xlab="X", ylab="Y",
     ylim=range(c(Y, condMean(x_prime, X, Y, mat5_2Kern, param))))

lines(x_prime, condMean(x_prime, X, Y, mat5_2Kern, param), col="red")

m = condMean(x_prime,X,Y,mat5_2Kern, param)
cov = condCov(x_prime,x_prime,X,mat5_2Kern,param)
int_sup = m + 1.96*sqrt(diag(cov))
int_inf = m - 1.96*sqrt(diag(cov))


