\name{print.mcmc.object}
\alias{print.mcmc.object}
\title{
  Print an rcppbugs object
}
\description{
  Print an rcppbugs object
}
\usage{
\method{print}{mcmc.object}(x)
}

\arguments{
  \item{x}{any rcppbugs object}
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
b <- normal.dist(rnorm(5),mu=0,tau=0.0001)
print(b)
}
\keyword{models}
