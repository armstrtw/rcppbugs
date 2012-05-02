\name{deterministic}
\alias{deterministic}
\alias{linear}
\alias{linear.grouped}
\alias{logistic}

\title{
  create deterministic rcppbugs objects
}
\description{
  Deterministic objects determine their values from other nodes in the
  mcmc model.  They have no contribution to the overall likelihood of
  the model, but are frequently used as hyperparmeters of stochastic
  objects.
}
\usage{
deterministic(f, ...)
linear(X,b)
linear.grouped(X,b,group)
}

\arguments{
  \item{f}{a user provided funtion that updates the value of the
    variable}
  \item{X}{the data matrix X to be used in estimation}
  \item{b}{the coefficient vector b to be used in estimation}
  \item{group}{an integer vector equal in length to the rows of X
  indicating the group membership of the corresponding row}
  \item{\dots}{arguments to function f}
}
\details{
  The number of arguments must match the arity of function f.
}
\value{
  an rcppbugs object representing the deterministic object.
}
\references{
https://github.com/armstrtw/CppBugs
}
\author{
rcppbugs was written by Whit Armstrong.
}
\seealso{
  \code{\link{logp},\link{run.model}}
}
\examples{
NR <- 1e2L
NC <- 2L
y <- matrix(rnorm(NR,1),nr=NR,nc=1L)
X <- matrix(nr=NR,nc=NC)

b <- mcmc.normal(rnorm(NC),mu=0,tau=0.0001)
y.hat.user <- deterministic(function(X,b) { X \%*\% b }, X, b)
y.hat.cpp <- linear(X,b)

}
\keyword{models}