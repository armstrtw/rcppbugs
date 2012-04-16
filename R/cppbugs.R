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

show.addr <- function(x) {
    invisible(.Call("getRawAddr",x,PACKAGE="RCppBugs"))
}

logp <- function(x) {
    .Call("logp",x,PACKAGE="RCppBugs")
}

jump <- function(x) {
    invisible(.Call("jump",x,PACKAGE="RCppBugs"))
}

print.mcmc <- function(x) {
    invisible(.Call("printMCMC",x,PACKAGE="RCppBugs"))
}

print.arma <- function(x) {
    invisible(.Call("printArma",x,PACKAGE="RCppBugs"))
}

create.model <- function(...) {
    .External("createModel",...,PACKAGE="RCppBugs")
}

run.model <- function(m, iterations, burn, adapt, thin) {
    .Call("run_model", m, iterations, burn, adapt, thin, PACKAGE="RCppBugs")
}

deterministic <- function(...) {
    .External("createDeterministic",...,PACKAGE="RCppBugs")
}

normal <- function(x,mu,tau,observed=FALSE) {
    .Call("createNormal",x,mu,tau,observed,PACKAGE="RCppBugs")
}

uniform <- function(x,lower,upper,observed=FALSE) {
    .Call("createUniform",x,lower,upper,observed,PACKAGE="RCppBugs")
}


get.history <- function(x) {
    .Call("getHist",x,PACKAGE="RCppBugs")
}
