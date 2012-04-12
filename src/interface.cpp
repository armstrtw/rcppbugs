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
#include <limits>
#include <stdexcept>
#include <boost/random.hpp>
#include <Rcpp.h>
#include <RcppArmadillo.h>
#include <cppbugs/cppbugs.hpp>
//#include "interface.h"
#include "helpers.h"
#include "raw.address.h"
#include "distribution.types.h"
#include "arma.context.h"
#include "assign.normal.logp.h"
#include "assign.uniform.logp.h"

typedef std::map<void*,ArmaContext*> contextMapT;

extern "C" SEXP logp(SEXP x);
extern "C" SEXP run_model(SEXP mcmc_nodes, SEXP iterations, SEXP burn_in, SEXP adapt, SEXP thin);
ArmaContext* mapToArma(contextMapT& m, SEXP x);
cppbugs::MCMCObject* createMCMC(contextMapT m, SEXP x);
cppbugs::MCMCObject* createNormal(contextMapT m, SEXP x);
cppbugs::MCMCObject* createUniform(contextMapT m, SEXP x);

SEXP logp(SEXP x) {
  double ans = std::numeric_limits<double>::quiet_NaN();
  contextMapT m;
  cppbugs::MCMCObject* node;
  try {
    node = createMCMC(m, x);
  } catch (std::logic_error &e) {
    REprintf("%s",e.what());
    return R_NilValue;
  }

  cppbugs::Stochastic* sp = dynamic_cast<cppbugs::Stochastic*>(node);
  if(sp && sp->getLikelihoodFunctor() ) {
    ans = sp->getLikelihoodFunctor()->calc();
  } else {
    REprintf("ERROR: could not convert node to stochastic");
  }
  return Rcpp::wrap(ans);
}

/*
SEXP run_model(SEXP nodes_sexp, SEXP iterations, SEXP burn_in, SEXP adapt, SEXP thin) {

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
      //mapToArma(contextMap,VECTOR_ELT(nodes_sexp,i));
      nodes.push_back(createMCMC(contextMap,VECTOR_ELT(nodes_sexp,i)));
    }
  } catch (std::logic_error &e) {
    REprintf("%s",e.what());
    return R_NilValue;
  }

  cppbugs::MCModel<boost::minstd_rand> m(nodes);
  m.sample(iterations_, burn_in_, adapt_, thin_);
  // walk the objects and append the histories as attributes
  return R_NilValue;
}
*/

ArmaContext* mapToArma(contextMapT& m, SEXP x) {
  //void* ptr = static_cast<void*>(RAW(x));
  void* ptr = rawAddress(x);

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

cppbugs::MCMCObject* createMCMC(contextMapT m, SEXP x) {
  SEXP distributed_sexp;
  distributed_sexp = Rf_getAttrib(x,Rf_install("distributed"));
  if(distributed_sexp == R_NilValue) {
    throw std::logic_error("ERROR: 'distributed' attribute not defined. Is this a stochastic variable?");
  }
  distT distributed = matchDistibution(std::string(CHAR(STRING_ELT(distributed_sexp,0))));

  cppbugs::MCMCObject* ans;

  switch(distributed) {
  case normalDistT:
    ans = createNormal(m, x);
    break;
  case uniformDistT:
    ans = createUniform(m, x);
    break;
  case gammaDistT:
  case betaDistT:
  case binomialDistT:
  default:
    ans = NULL;
    throw std::logic_error("ERROR: distribution not supported yet.");
  }
  return ans;
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

cppbugs::MCMCObject* createUniform(contextMapT m, SEXP x) {
  cppbugs::MCMCObject* ans;

  SEXP lower_sexp = Rf_getAttrib(x,Rf_install("lower"));
  SEXP upper_sexp = Rf_getAttrib(x,Rf_install("upper"));
  SEXP observed_sexp = Rf_getAttrib(x,Rf_install("observed"));

  if(lower_sexp == R_NilValue || upper_sexp == R_NilValue || observed_sexp == R_NilValue) {
    throw std::logic_error("ERROR: needed attribute not defined.");
  }

  //bool observed = LOGICAL(observed_sexp)[0];
  bool observed = Rcpp::as<bool>(observed_sexp);

  // map to arma types
  ArmaContext* x_arma = mapToArma(m, x);
  ArmaContext* lower_arma = mapToArma(m, lower_sexp);
  ArmaContext* upper_arma = mapToArma(m, upper_sexp);

  switch(x_arma->getArmaType()) {
  case doubleT:
    if(observed) {
      ans = assignUniformLogp<cppbugs::ObservedUniform>(x_arma->getDouble(),lower_arma,upper_arma);
    } else {
      ans = assignUniformLogp<cppbugs::Uniform>(x_arma->getDouble(),lower_arma,upper_arma);
    }
    break;
  case vecT:
    if(observed) {
      ans = assignUniformLogp<cppbugs::ObservedUniform>(x_arma->getVec(),lower_arma,upper_arma);
    } else {
      ans = assignUniformLogp<cppbugs::Uniform>(x_arma->getVec(),lower_arma,upper_arma);
    }
    break;
  case matT:
    if(observed) {
      ans = assignUniformLogp<cppbugs::ObservedUniform>(x_arma->getMat(),lower_arma,upper_arma);
    } else {
      ans = assignUniformLogp<cppbugs::Uniform>(x_arma->getMat(),lower_arma,upper_arma);
    }
    break;
  case intT:
  case ivecT:
  case imatT:
  default:
    throw std::logic_error("ERROR: uniform must be a continuous variable type (double, vec, or mat).");
  }
  return ans;
}
