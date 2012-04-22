###########################################################################
## Copyright (C) 2012  Whit Armstrong                                    ##
##                                                                       ##
## This program is free software: you can redistribute it and#or modify  ##
## it under the terms of the GNU General Public License as published by  ##
## the Free Software Foundation, either version 3 of the License, or     ##
## (at your option) any later version.                                   ##
##                                                                       ##
## This program is distributed in the hope that it will be useful,       ##
## but WITHOUT ANY WARRANTY; without even the implied warranty of        ##
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         ##
## GNU General Public License for more details.                          ##
##                                                                       ##
## You should have received a copy of the GNU General Public License     ##
## along with this program.  If not, see <http:##www.gnu.org#licenses#>. ##
###########################################################################

print.mcmc.object <- function(x,digits = NULL, quote = TRUE, na.print = NULL, print.gap = NULL,
    right = FALSE, max = NULL, useSource = TRUE, ...) {

    x <- unclass(x)
    xd <- dim(x)
    attributes(x) <- NULL
    if(!is.null(xd)) {
        dim(x) <- xd
    }
    print.default(x,digits, quote, na.print, print.gap, right, max, useSource, ...)
}

logp <- function(x) {
    .Call("logp",x,PACKAGE="rcppbugs")
}

## print.mcmc <- function(x) {
##     invisible(.Call("printMCMC",x,PACKAGE="rcppbugs"))
## }

## print.arma <- function(x) {
##     invisible(.Call("printArma",x,PACKAGE="rcppbugs"))
## }

create.model <- function(...) {
    m <- match.call()
    attr(m,"env") <- new.env()
    class(m) <- "mcmc.model"
    m
}

run.model <- function(m, iterations, burn, adapt, thin) {
    .Call("runModel", m, iterations, burn, adapt, thin, PACKAGE="rcppbugs")
}

deterministic <- function(f,...) {
    mc <- match.call()
    stopifnot(typeof(eval(mc[[2]]))=="closure")
    ## capture shape/type of result
    x <- do.call(f,list(...))
    attr(x,"distributed") <- "deterministic"
    attr(x,"update.method") <- compiler::compile(f)
    attr(x,"call") <- mc
    attr(x,"env") <- new.env()
    class(x) <- "mcmc.object"
    x
}

linear <- function(X,b) {
    stopifnot(is.null(dim(b)))
    stopifnot(!is.null(dim(X)))
    stopifnot(length(dim(X))==2L)
    x <- X %*% b
    attr(x,"distributed") <- "linear.deterministic"
    attr(x,"X") <- substitute(X)
    attr(x,"b") <- substitute(b)
    attr(x,"env") <- new.env()
    class(x) <- "mcmc.object"
    x
}


logistic <- function(X,b) {
    stopifnot(is.null(dim(b)))
    stopifnot(!is.null(dim(X)))
    stopifnot(length(dim(X))==2L)
    x <- 1/(1 + exp(-X %*% b))
    attr(x,"distributed") <- "logistic.deterministic"
    attr(x,"X") <- substitute(X)
    attr(x,"b") <- substitute(b)
    attr(x,"env") <- new.env()
    class(x) <- "mcmc.object"
    x
}

mcmc.normal <- function(x,mu,tau,observed=FALSE) {
    attr(x,"distributed") <- "normal"
    attr(x,"mu") <- substitute(mu)
    attr(x,"tau") <- substitute(tau)
    attr(x,"observed") <- observed
    attr(x,"env") <- new.env()
    class(x) <- "mcmc.object"
    x
}

mcmc.uniform <- function(x,lower,upper,observed=FALSE) {
    attr(x,"distributed") <- "uniform"
    attr(x,"lower") <- substitute(lower)
    attr(x,"upper") <- substitute(upper)
    attr(x,"observed") <- observed
    attr(x,"env") <- new.env()
    class(x) <- "mcmc.object"
    x
}

mcmc.gamma <- function(x,alpha,beta,observed=FALSE) {
    attr(x,"distributed") <- "gamma"
    attr(x,"alpha") <- substitute(alpha)
    attr(x,"beta") <- substitute(beta)
    attr(x,"observed") <- observed
    attr(x,"env") <- new.env()
    class(x) <- "mcmc.object"
    x
}

mcmc.bernoulli <- function(x,p,observed=FALSE) {
    attr(x,"distributed") <- "bernoulli"
    attr(x,"p") <- substitute(p)
    attr(x,"observed") <- observed
    attr(x,"env") <- new.env()
    class(x) <- "mcmc.object"
    x
}

mcmc.binomial <- function(x,n,p,observed=FALSE) {
    attr(x,"distributed") <- "binomial"
    attr(x,"n") <- substitute(n)
    attr(x,"p") <- substitute(p)
    attr(x,"observed") <- observed
    attr(x,"env") <- new.env()
    class(x) <- "mcmc.object"
    x
}
