\name{run.model}
\alias{run.model}
\alias{create.model}
\title{
  Create and run rcppbugs models.
}
\description{
  The create.model function creates a model object.
  The run.model function simulates an mcmc chain of the nodes in the
  model.
  After the completion of run.model, the history of the run is returned
  as a named list.
}
\usage{
create.model(...)
run.model(m, iterations, burn, adapt, thin)
}

\arguments{
  \item{m}{the rcppbugs model object.}
  \item{iterations}{how many iterations to sample.}
  \item{burn}{how many iterations to use for burnin.}
  \item{adapt}{how many iterations to use for the adaptive period.}
  \item{thin}{how frequently to record traces of the model nodes.}
  \item{\dots}{rcppbugs objects to use as the nodes of the model}
}
\value{
  create.model returns a mcmc.model model object.
  run.model returns a named list containing the historical traces of the
  model run.
}
\references{
https://github.com/armstrtw/CppBugs
}
\author{
rcppbugs was written by Whit Armstrong.
}

\seealso{
  \code{\link{logp}}
}
\examples{
library(rcppbugs)


NR <- 1e2L
NC <- 2L
y <- matrix(rnorm(NR,1) + 10,nr=NR,nc=1L)
X <- matrix(nr=NR,nc=NC)
X[,1] <- 1
X[,2] <- y + rnorm(NR)/2 - 10

## RCppBugs Model
b <- mcmc.normal(rnorm(NC),mu=0,tau=0.0001)
tau.y <- mcmc.gamma(sd(as.vector(y)),alpha=0.1,beta=0.1)
y.hat <- linear(X,b)
y.lik <- mcmc.normal(y,mu=y.hat,tau=tau.y,observed=TRUE)
m <- create.model(b, tau.y, y.hat, y.lik)


runtime <- system.time(ans <- run.model(m, iterations=1e2L, burn=1e2L, adapt=1e3L, thin=10L))
print(apply(ans[["b"]],2,mean))

}
\keyword{models}
