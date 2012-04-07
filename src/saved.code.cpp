
MCMCObject* createNodePtr(SEXP x, SEXP rho) {
  MCMCObject* ans(NULL);

  // cppbugs distribution
  std::string distributed(getAttr("distributed"));

  // only look at the classname in the position 0
  std::string class0(CHAR(STRING_ELT(getAttrib(x,R_ClassSymbol),0)));

  if(class0=="function") {
    SEXP tmp_eval;
    PROTECT(tmp_eval= eval(x, rho));
    if(isMatrix(tmp_eval)) {
      ans = new RDeterministic<arma::mat>(x, rho);
    } else {
      ans = new RDeterministic<arma::vec>(x, rho);
    }
    UNPROTECT(1);
    return ans;
  }

  // FIXME: if(distributed.size() == 0) { throw }

  if(distributed=="normal") {
    if(isMatrix(x)) {
      ans = new Normal<arma::mat>(Rcpp::as<arma::mat>(x));
    } else {
      ans = new Normal<arma::vec>(Rcpp::as<arma::vec>(x));
    }
  } else if(distributed=="uniform") {
    if(isMatrix(x)) {
      ans = new Uniform<arma::mat>(Rcpp::as<arma::mat>(x));
    } else {
      ans = new Uniform<arma::vec>(Rcpp::as<arma::vec>(x));
    }
  } else {
    // throw
  }
  return ans;
}



// double evaluate(long *l_nfeval, SEXP par, SEXP fcall, SEXP env)
// {
//    SEXP sexp_fvec, fn;
//    double f_result;

//    PROTECT(fn = lang3(fcall, par, R_DotsSymbol));
//       (*l_nfeval)++;  /* increment function evaluation count */

//    PROTECT(sexp_fvec = eval(fn, env));
//    f_result = NUMERIC_POINTER(sexp_fvec)[0];

//    if(ISNAN(f_result))
//      error("NaN value of objective function! \nPerhaps adjust the bounds.");

//    UNPROTECT(2);
//    return(f_result);
// }
