///////////////////////////////////////////////////////////////////////////
// Copyright (C) 2011  Whit Armstrong                                    //
//                                                                       //
// This program is free software: you can redistribute it and/or modify  //
// it under the terms of the GNU General Public License as published by  //
// the Free Software Foundation, either version 3 of the License, or     //
// (at your option) any later version.                                   //
//                                                                       //
// This program is distributed in the hope that it will be useful,       //
// but WITHOUT ANY WARRANTY; without even the implied warranty of        //
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         //
// GNU General Public License for more details.                          //
//                                                                       //
// You should have received a copy of the GNU General Public License     //
// along with this program.  If not, see <http://www.gnu.org/licenses/>. //
///////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <Rcpp.h>
#include <RcppArmadillo.h>
#include "interface.hpp"
using namespace arma;


MCMCObject* createNodePtr(SEXP x, SEXP rho);
std::string getAttr(SEXP x, const char* attr_name);
bool isMatrix(SEXP x);
void mapToArma(SEXP x);
std::vector<R_len_t> getDims(SEXP x);
void setAtty(SEXP x, const char* atty_name, const char* str);
void setAtty(SEXP x, const char* atty_name, SEXP sexp);

void setAtty(SEXP x, const char* atty_name, const char* str) {
  PROTECT(atty = allocVector(STRSXP, 1));
  SET_STRING_ELT(atty, 0, mkChar(str));
  setAttrib(x, install(atty_name), str);
  UNPROTECT(1);
}

void setAtty(SEXP x, const char* atty_name, SEXP sexp) {
  PROTECT(atty = allocVector(STRSXP, 1));
  SET_STRING_ELT(atty, 0, mkChar(str));
  setAttrib(x, install(atty_name), sexp);
  UNPROTECT(1);
}

void mapToArma(SEXP x) {
  SEXP arma_ptr_sexp;
  switch(TYPEOF(x)) {
  case REALSXP:
    switch(getDims(x).size()) {
    case 0:
      // add case for scalar values
    case 1:
      PROTECT(arma_ptr_sexp = R_MakeExternalPtr(reinterpret_cast<void*>(new Rcpp::as<arma::vec>(x)),install("arma_ptr"),R_NilValue));
      //FIXME: register finalizer
      setAtty(x,"arma_ptr", arma_ptr_sexp);
      setAtty(x,"arma_type","vec");
      break;
    case 2:
      PROTECT(arma_ptr_sexp = R_MakeExternalPtr(reinterpret_cast<void*>(new Rcpp::as<arma::mat>(x)),install("arma_ptr"),R_NilValue));
      //FIXME: register finalizer
      setAtty(x,"arma_ptr", arma_ptr_sexp);
      setAtty(x,"arma_type","mat");
      break;
    default:
      Rprintf("ERROR: tensor conversion not supported yet.");
    }
    break;
  case LGLSXP:
  case INTSXP:
    switch(getDims(x).size()) {
    case 0:
      // add case for scalar values
    case 1:
      PROTECT(arma_ptr_sexp = R_MakeExternalPtr(reinterpret_cast<void*>(new Rcpp::as<arma::ivec>(x)),install("arma_ptr"),R_NilValue));
      //FIXME: register finalizer
      setAtty(x,"arma_ptr", arma_ptr_sexp);
      setAtty(x,"arma_type","ivec");
      break;
    case 2:
      PROTECT(arma_ptr_sexp = R_MakeExternalPtr(reinterpret_cast<void*>(new Rcpp::as<arma::imat>(x)),install("arma_ptr"),R_NilValue));
      //FIXME: register finalizer
      setAtty(x,"arma_ptr", arma_ptr_sexp);
      setAtty(x,"arma_type","imat");
      break;
    default:
      Rprintf("ERROR: tensor conversion not supported yet.");
    }
    break;
  default:
    Rprintf("ERROR: conversion not supported.");
  }
}

std::string getAttr(SEXP x, const char* attr_name) {
  std::string ans;
  SEXP attr = getAttrib(x,install(attr_name));
  if(attr != R_NilValue) {
    ans = std::string(CHAR(STRING_ELT(attr,0)));
  }
  return ans;
}

bool isMatrix(SEXP x) {
  return getAttrib(x,R_DimSymbol) != R_NilValue ? true : false;
}

std::vector<R_len_t> getDims(SEXP x) {
  std::vector<R_len_t> ans;
  SEXP dims = getAttrib(x, R_DimSymbol);
  if(dims == R_NilValue) {
    ans.push_back(LENGTH(x));
  } else {
    for(R_len_t i = 0; i < LENGTH(dims)) {
      ans.push_back(INTEGER(dims)[i]);
    }
  }
  return ans;
}

SEXP run_model(SEXP mcmc_nodes, SEXP iterations, SEXP burn_in, SEXP adapt, SEXP thin, SEXP rho) {

  int iterations_ = Rcpp::as<int>(iterations);
  int burn_in_ = Rcpp::as<int>(burn_in);
  int adapt_ = Rcpp::as<int>(adapt);
  int thin_ = Rcpp::as<int>(thin);

  std::vector<MCMCObject*> nodes;

  // attempt to catch object conversion errors
  // i.e. mismatches between R types and arma types
  try {
    for(R_len_t i = 0; i < length(mcmc_nodes_); i++) {
      nodes.push_back(createNodePtr(VECTOR_ELT(nodes,i),rho));
    }
  } catch (std::logic_error &e) {
    cout << e.what() << endl;
  };

  MCModel m(nodes);
  m.sample(iterations_, burn_in_, adapt_, thin_);
  // walk the objects and append the histories as attributes
  return R_NilValue;
}


MCMCObject* createNormal(SEXP x) {
  MCMCObject* ans;
  std::vector<R_len_t> dims = getDims(x);
  SEXP mu_sexp = getAttrib(x,install("mu"));
  SEXP tau_sexp = getAttrib(x,install("tau"));
  SEXP observed_sexp = getAttrib(x,install("observed"));

  if(mu_sexp == R_NilValue || tau_sexp == R_NilValue || observed_sexp == R_NilValue) {
    throw logic_error("ERROR: needed attribute not defined.");
  }

  //bool observed = LOGICAL(observed_sexp)[0];
  bool observed = Rcpp::as<bool>(observed_sexp);

  // normal must be real valued
  switch(dims.size()) {
  case 0:
    if(observed) {
      ans = new ObservedNormal<double>(reinterpret_cast<double*>(getAttrib(x,install("arma_ptr"))));
    } else {
      ans = new Normal<double>(reinterpret_cast<double*>(getAttrib(x,install("arma_ptr"))));
    }
    break;
  case 1:
    if(observed) {
      ans = new ObservedNormal<vec>(reinterpret_cast<vec*>(getAttrib(x,install("arma_ptr"))));
    } else {
      ans = new Normal<vec>(reinterpret_cast<vec*>(getAttrib(x,install("arma_ptr"))));
    }
    break;
  case 2:
    if(observed) {
      ans = new ObservedNormal<mat>(reinterpret_cast<mat*>(getAttrib(x,install("arma_ptr"))));
    } else {
      ans = new Normal<mat>(reinterpret_cast<mat*>(getAttrib(x,install("arma_ptr"))));
    }
    break;
  }
  assignNormalLogp(ans,mu_sexp,tau_sexp);
}
