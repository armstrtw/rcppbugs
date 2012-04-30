\name{mcmc.distributions}
\alias{mcmc.normal}
\alias{mcmc.uniform}
\alias{mcmc.gamma}
\alias{mcmc.beta}
\alias{mcmc.bernoulli}
\alias{mcmc.binomial}

\title{
  Create a stochastic rcppbugs object.
}
\description{
  Create stochastic objects in the spirit of PyMC.
}
\usage{
mcmc.normal(x, mu, tau, observed = FALSE)
mcmc.uniform(x, lower, upper, observed = FALSE)
mcmc.gamma(x, alpha, beta, observed = FALSE)
mcmc.beta(x, alpha, beta, observed = FALSE)
mcmc.bernoulli(x, p, observed = FALSE)
mcmc.binomial(x, n, p, observed = FALSE)
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

b <- mcmc.normal(rnorm(10),mu=0,tau=0.0001)
tau <- mcmc.gamma(runif(1),alpha=0.1,beta=0.1)
b.unif <- mcmc.uniform(runif(1),lower=0,upper=100)

}
\keyword{models}

