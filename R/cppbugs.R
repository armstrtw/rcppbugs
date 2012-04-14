###########################################################################
## Copyright (C) 2011  Whit Armstrong                                    ##
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

show.addr <- function(x) {
    .Call("getRawAddr",x,PACKAGE="RCppBugs")
}

attach.args <- function(...) {
    .External("attachArgs",...,PACKAGE="RCppBugs")
}

logp <- function(x) {
    .Call("logp",x,PACKAGE="RCppBugs")
}

create.model <- function(...) {
    .External("createModel",...,PACKAGE="RCppBugs")
}

run.model <- function(m, iterations, burn, adapt, thin) {
    .Call("run_model", m, iterations, burn, adapt, thin, PACKAGE="RCppBugs")
}

deterministic <- function(f,...) {
    args <- list(...)

    ## this could fail for internl/primitive functions
    stopifnot(length(formals(f))==length(args))
    x <- do.call(f,args)
    attr(x,"distributed") <- "deterministic"
    attr(x,"fun") <- f
    attach.args(x,...)
    x
}

normal <- function(x,mu,tau,observed=FALSE) {
    attr(x,"distributed") <- "normal"
    attr(x,"mu") <- mu
    attr(x,"tau") <- tau
    attr(x,"observed") <- observed
    x
}

uniform <- function(x,lower,upper,observed=FALSE) {
    attr(x,"distributed") <- "uniform"
    attr(x,"lower") <- lower
    attr(x,"upper") <- upper
    attr(x,"observed") <- observed
    x
}
