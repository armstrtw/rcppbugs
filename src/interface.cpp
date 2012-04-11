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

#include <map>
#include <stdexcept>
#include <boost/random.hpp>
#include <cppbugs/cppbugs.hpp>
#include <Rcpp.h>
#include <RcppArmadillo.h>
//#include "interface.h"
#include "helpers.h"
#include "distribution.types.h"
#include "arma.context.h"
#include "assign.normal.logp.h"

typedef std::map<void*,ArmaContext*> contextMapT;

extern "C" SEXP run_model(SEXP mcmc_nodes, SEXP iterations, SEXP burn_in, SEXP adapt, SEXP thin, SEXP rho);
cppbugs::MCMCObject* createNodePtr(SEXP x, SEXP rho);
void mapToArma(SEXP x);
void assignNormalLogp(cppbugs::MCMCObject* x, ArmaContext* mu, ArmaContext* tau);

ArmaContext* mapToArma(contextMapT& m, SEXP x) {
  void* ptr = static_cast<void*>(RAW(x));

  // if node is mapped, just return mapped pointer
  if(!m.count(ptr)) {
    switch(TYPEOF(x)) {
    case REALSXP:
      switch(getDims(x).size()) {
      case 0: m[ptr] = new ArmaDouble(x); break;
      case 1: m[ptr] = new ArmaVec(x); break;
      case 2: m[ptr] = new ArmaMat(x); break;
      default:
        throw std::logic_error("ERROR: tensor conversion not supported yet.");
      }
      break;
    case LGLSXP:
    case INTSXP:
      throw std::logic_error("ERROR: integer type conversion not supported yet.");
      break;
    default:
      throw std::logic_error("ERROR: conversion not supported.");
    }
  }
  return m[ptr];
}

SEXP run_model(SEXP nodes_sexp, SEXP iterations, SEXP burn_in, SEXP adapt, SEXP thin, SEXP rho) {

  int iterations_ = Rcpp::as<int>(iterations);
  int burn_in_ = Rcpp::as<int>(burn_in);
  int adapt_ = Rcpp::as<int>(adapt);
  int thin_ = Rcpp::as<int>(thin);

  contextMapT contextMap;
  std::vector<cppbugs::MCMCObject*> nodes;

  // attempt to catch object conversion errors
  // i.e. mismatches between R types and arma types
  try {
    for(R_len_t i = 0; i < Rf_length(nodes_sexp); i++) {
      //nodes.push_back(createNodePtr(VECTOR_ELT(nodes_sexp,i),rho));
      mapToArma(contextMap,VECTOR_ELT(nodes_sexp,i));
    }
  } catch (std::logic_error &e) {
    Rprintf("%s",e.what());
    return R_NilValue;
  }

  // ** distT stringIndex(const std::string distibution) **
  cppbugs::MCModel<boost::minstd_rand> m(nodes);
  m.sample(iterations_, burn_in_, adapt_, thin_);
  // walk the objects and append the histories as attributes
  return R_NilValue;
}

cppbugs::MCMCObject* createNormal(contextMapT m, SEXP x) {
  cppbugs::MCMCObject* ans;

  SEXP mu_sexp = Rf_getAttrib(x,Rf_install("mu"));
  SEXP tau_sexp = Rf_getAttrib(x,Rf_install("tau"));
  SEXP observed_sexp = Rf_getAttrib(x,Rf_install("observed"));

  if(mu_sexp == R_NilValue || tau_sexp == R_NilValue || observed_sexp == R_NilValue) {
    throw std::logic_error("ERROR: needed attribute not defined.");
  }

  //bool observed = LOGICAL(observed_sexp)[0];
  bool observed = Rcpp::as<bool>(observed_sexp);

  // map to arma types
  ArmaContext* x_arma = mapToArma(m, x);
  ArmaContext* mu_arma = mapToArma(m, mu_sexp);
  ArmaContext* tau_arma = mapToArma(m, tau_sexp);

  switch(x_arma->getArmaType()) {
  case doubleT:
    if(observed) {
      ans = assignNormalLogp<cppbugs::ObservedNormal>(x_arma->getDouble(),mu_arma,tau_arma);
    } else {
      ans = assignNormalLogp<cppbugs::Normal>(x_arma->getDouble(),mu_arma,tau_arma);
    }
    break;
  case vecT:
    if(observed) {
      ans = assignNormalLogp<cppbugs::ObservedNormal>(x_arma->getVec(),mu_arma,tau_arma);
    } else {
      ans = assignNormalLogp<cppbugs::Normal>(x_arma->getVec(),mu_arma,tau_arma);
    }
    break;
  case matT:
    if(observed) {
      ans = assignNormalLogp<cppbugs::ObservedNormal>(x_arma->getMat(),mu_arma,tau_arma);
    } else {
      ans = assignNormalLogp<cppbugs::Normal>(x_arma->getMat(),mu_arma,tau_arma);
    }
    break;
  case intT:
  case ivecT:
  case imatT:
  default:
    throw std::logic_error("ERROR: normal must be a continuous variable type (double, vec, or mat).");
  }
  return ans;
}
