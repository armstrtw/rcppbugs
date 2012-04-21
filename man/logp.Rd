\name{logp}
\alias{logp}
\title{
  Calculate the log likelihood of an rcppbugs object.
}
\description{
  Calculate the log likelihood of an rcppbugs object
  based on the current state of the dependant(upstream)
  objects in the chain.
}
\usage{
logp(x)
}

\arguments{
  \item{x}{any stochastic rcppbugs object}
}
\value{
  a double precision value indicating the total log likelihood of the object.
}
\references{
https://github.com/armstrtw/CppBugs
}
\author{
rcppbugs was written by Whit Armstrong.
}

\seealso{
  \code{\link{normal.dist}}
}
\examples{
NR <- 100
NC <- 2

b <- normal.dist(rnorm(NC),mu=0,tau=0.0001)
tau.y <- gamma.dist(runif(1),alpha=0.1,beta=0.1)

logp(b)
logp(tau.y)


}
\keyword{models}
