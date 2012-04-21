************
Introduction
************

:Date: April 21, 2012
:Authors: Whit Armstrong
:Contact: armstrong.whit@gmail.com
:Web site: http://github.com/armstrtw/rcppbugs
:License: GPL-3


Purpose
=======

rcppbugs is an interface to the CppBugs c++ library designed for MCMC sampling.


Features
========

rcppbugs attempts to make writing mcmc models as painless as possible.  It incorporates features
from both WinBugs and PyMC. Users define normal R objects to represent the nodes of their model.  Deterministic nodes are represented by R functions with an accompanying argument list.

* rcppbugs makes heavey use of Armadillo (http://arma.sourceforge.net/) which allows for very fast matrix algebra.  Heavy use of templates allows the code to be very generic and easily extended. Basic statistical distributions are supported in this release and many more will be implemented. Eventually the packages should be as feature complete as WinBugs and PyMC. 


Usage
=====

Here is a simple example of a linear model in rcppbugs.

* define the variable space

* implement a function which updates the deterministic variables

* add nodes to the model and define the parameters governing the stochastic variables

::

	library(rcppbugs)
	
	
	## set up the test data
	NR <- 1e2L
	NC <- 2L
	y <- matrix(rnorm(NR,1) + 10,nr=NR,nc=1L)
	X <- matrix(nr=NR,nc=NC)
	X[,1] <- 1
	X[,2] <- y + rnorm(NR)/2 - 10
	
	## run a normal linear model w/ lm.fit to check results
	lm.res <- lm.fit(X,y)
	print(coef(lm.res))
	
	## RCppBugs Model
	
	b <- normal.dist(rnorm(NC),mu=0,tau=0.0001)
	tau.y <- gamma.dist(sd(as.vector(y)),alpha=0.1,beta=0.1)
	
	## can also define a user supplied deterministic function
	## y.hat <- deterministic(function(X,b) { X %*% b }, X, b)
	
	y.hat <- linear(X,b)
	y.lik <- normal.dist(y,mu=y.hat,tau=tau.y,observed=TRUE)
	m <- create.model(b, tau.y, y.hat, y.lik)
	
	cat("running model...\n")
	runtime <- system.time(ans <- run.model(m, iterations=1e5L, burn=1e4L, adapt=1e3L, thin=10L))
	print(apply(ans[["b"]],2,mean))
	
	print(runtime)
	

