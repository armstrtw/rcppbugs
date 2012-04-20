.onAttach <- function(libname, pkgname) {
    ## force R to init the rng
    ## otherwise if you attempt to draw rng from c++
    ## the rng will not generate new variates (wtf!)
    rnorm(1)
}
