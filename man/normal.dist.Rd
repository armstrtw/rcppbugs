\name{normal.dist}
\alias{normal.dist}
\alias{uniform.dist}
\alias{gamma.dist}
\alias{bernoulli.dist}
\alias{binomial.dist}

\title{
  Create a stochastic rcppbugs object.
}
\description{
  Create stochastic objects in the spirit of PyMC.
}
\usage{
normal.dist(x, mu, tau, observed = FALSE)
uniform.dist(x, lower, upper, observed = FALSE)
gamma.dist(x, alpha, beta, observed = FALSE)
bernoulli.dist(x, p, observed = FALSE)
binomial.dist(x, n, p, observed = FALSE)
}

\arguments{
  \item{x}{the initial value of the object}

  \item{mu}{the mean for normally distributed objects}
  \item{tau}{the precision of normally distributed objects}

  \item{lower}{the lower limit of the uniform distribution}
  \item{upper}{the upper limit of the uniform distribution}

  \item{alpha}{the shape parameter of the gamma distribution}
  \item{beta}{the scale parameter of the gamma distribution}

  \item{n}{the sample size of a binomial distribution}
  \item{p}{the success rate in bernoulli or binomial distributions}

  
  \item{observed}{whether the object should be treated as constant data
    or simulated over the MCMC chain}
}
\value{
  an rcppbugs object corresponding to the particular distribution requested
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
NR <- 100
NC <- 2
y <- matrix(rnorm(NR,1),nr=NR,nc=1L)
b <- normal.dist(rnorm(NC),mu=0,tau=0.0001)
tau.y <- gamma.dist(sd(as.vector(y)),alpha=0.1,beta=0.1)
y.lik <- normal.dist(y,mu=y.hat,tau=tau.y,observed=TRUE)
}
\keyword{models}

