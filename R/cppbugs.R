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

print.mcmc.object <- function(x) {

    x <- unclass(x)
    xd <- dim(x)
    attributes(x) <- NULL
    if(!is.null(xd)) {
        dim(x) <- xd
    }
    print.default(x)
}

logp <- function(x) {
    .Call("logp",x,PACKAGE="RCppBugs")
}

jump <- function(x) {
    invisible(.Call("jump",x,PACKAGE="RCppBugs"))
}

## print.mcmc <- function(x) {
##     invisible(.Call("printMCMC",x,PACKAGE="RCppBugs"))
## }

## print.arma <- function(x) {
##     invisible(.Call("printArma",x,PACKAGE="RCppBugs"))
## }

create.model <- function(...) {
    m <- match.call()
    attr(m,"env") <- new.env()
    class(m) <- "mcmc.model"
    m
}

run.model <- function(m, iterations, burn, adapt, thin) {
    .Call("runModel", m, iterations, burn, adapt, thin, PACKAGE="RCppBugs")
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

normal <- function(x,mu,tau,observed=FALSE) {
    attr(x,"distributed") <- "normal"
    attr(x,"mu") <- substitute(mu)
    attr(x,"tau") <- substitute(tau)
    attr(x,"observed") <- observed
    attr(x,"env") <- new.env()
    class(x) <- "mcmc.object"
    x
}

uniform <- function(x,lower,upper,observed=FALSE) {
    attr(x,"distributed") <- "uniform"
    attr(x,"lower") <- substitute(lower)
    attr(x,"upper") <- substitute(upper)
    attr(x,"observed") <- observed
    attr(x,"env") <- new.env()
    class(x) <- "mcmc.object"
    x
}
