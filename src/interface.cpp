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
#include <map>
#include <limits>
#include <stdexcept>
#include <boost/random.hpp>
#include <Rcpp.h>
#include <RcppArmadillo.h>
//#include <cppbugs/cppbugs.hpp>
#include <cppbugs/mcmc.deterministic.hpp>
#include <cppbugs/mcmc.normal.hpp>
#include <cppbugs/mcmc.uniform.hpp>
#include <cppbugs/mcmc.gamma.hpp>
#include <cppbugs/mcmc.binomial.hpp>
#include <cppbugs/mcmc.bernoulli.hpp>

#include "helpers.h"
#include "finalizers.h"
#include "raw.address.h"
#include "distribution.types.h"
#include "arma.context.h"
#include "assign.normal.logp.h"
#include "assign.uniform.logp.h"
#include "r.deterministic.h"
#include "r.mcmc.model.h"

typedef std::map<void*,ArmaContext*> vpArmaMapT;
typedef std::map<void*,cppbugs::MCMCObject*> vpMCMCMapT;

// public interface
extern "C" SEXP logp(SEXP x);
extern "C" SEXP jump(SEXP x);
extern "C" SEXP createModel(SEXP args_sexp);
extern "C" SEXP runModel(SEXP mp_, SEXP iterations, SEXP burn_in, SEXP adapt, SEXP thin);

// private methods
cppbugs::MCMCObject* createMCMC(SEXP x, vpArmaMapT& armaMap);
cppbugs::MCMCObject* createDeterministic(SEXP args_, vpArmaMapT& armaMap);
cppbugs::MCMCObject* createNormal(SEXP x_, vpArmaMapT& armaMap);
cppbugs::MCMCObject* createUniform(SEXP x_, vpArmaMapT& armaMap);
ArmaContext* getArma(SEXP x);
void initArgList(SEXP args, arglistT& arglist, const size_t skip);
SEXP makeNames(std::vector<const char*>& argnames);
SEXP createTrace(arglistT& arglist, vpArmaMapT& armaMap, vpMCMCMapT& mcmcMap);

void initArgList(SEXP args, arglistT& arglist, const size_t skip) {

  for(size_t i = 0; i < skip; i++) {
    args = CDR(args);
  }

  // loop through rest of args
  for(; args != R_NilValue; args = CDR(args)) {
    arglist.push_back(CAR(args));
  }
}

SEXP logp(SEXP x_) {
  double ans = std::numeric_limits<double>::quiet_NaN();
  cppbugs::MCMCObject* node(NULL);
  vpArmaMapT armaMap;
  try {
    ArmaContext* ap = getArma(x_);
    armaMap[rawAddress(x_)] = ap;
    node = createMCMC(x_,armaMap);
  } catch (std::logic_error &e) {
    REprintf("%s\n",e.what());
    return R_NilValue;
  }

  cppbugs::Stochastic* sp = dynamic_cast<cppbugs::Stochastic*>(node);
  if(sp) {
    ans = sp->loglik();
  } else {
    REprintf("ERROR: could not convert node to stochastic.\n");
  }
  return Rcpp::wrap(ans);
}

SEXP jump(SEXP x_) {
  static cppbugs::RNativeRng rng;
  cppbugs::MCMCObject* node(NULL);
  vpArmaMapT armaMap;

  try {
    ArmaContext* ap = getArma(x_);
    armaMap[rawAddress(x_)] = ap;
    node = createMCMC(x_,armaMap);
  } catch (std::logic_error &e) {
    REprintf("%s\n",e.what());
    return R_NilValue;
  }
  node->jump(rng);
  return R_NilValue;
}

template<typename T>
SEXP getHistory(cppbugs::MCMCObject* node) {
  //SEXP ans;
  cppbugs::MCMCSpecialized<T>* sp = dynamic_cast<cppbugs::MCMCSpecialized<T>*>(node);
  if(sp == nullptr) {
    throw std::logic_error("invalid node conversion.");
  }
  //Rprintf("getHistory<T> history.size(): %d\n",sp->history.size());
  //PROTECT(ans = Rf_allocVector(VECSXP, sp->history.size()));
  //Rcpp::List ans(sp->history.size());
  //const size_t NC = sp->history.begin()->n_elem;
  const size_t NC = sp->history.begin()->n_cols;
  Rcpp::NumericMatrix ans(sp->history.size(),NC);
  R_len_t i = 0;
  for(typename std::list<T>::const_iterator it = sp->history.begin(); it != sp->history.end(); it++) {
    //SET_VECTOR_ELT(ans, i, Rcpp::wrap(*it)); ++i;
    //ans[i] = Rcpp::wrap(*it);
    //Rprintf("%d %d",i, it->n_cols);
    for(size_t j = 0; j < NC; j++) {
      ans(i,j) = it->at(j);
    }
    ++i;
  }
  //UNPROTECT(1);
  return Rcpp::wrap(ans);
}

template<> SEXP getHistory<arma::vec>(cppbugs::MCMCObject* node) {
  //SEXP ans;
  cppbugs::MCMCSpecialized<arma::vec>* sp = dynamic_cast<cppbugs::MCMCSpecialized<arma::vec>*>(node);
  if(sp == nullptr) {
    throw std::logic_error("invalid node conversion.");
  }

  if(sp->history.size()==0) {
    return R_NilValue;
  }

  //Rprintf("getHistory<arma::vec> history.size(): %d\n",sp->history.size());
  const size_t NC = sp->history.begin()->n_elem;
  //Rprintf("getHistory<arma::vec> history dim: %d\n",NC);
  Rcpp::NumericMatrix ans(sp->history.size(),NC);
  R_len_t i = 0;
  for(typename std::list<arma::vec>::const_iterator it = sp->history.begin(); it != sp->history.end(); it++) {
    for(size_t j = 0; j < NC; j++) {
      ans(i,j) = it->at(j);
    }
    ++i;
  }
  //UNPROTECT(1);
  return Rcpp::wrap(ans);
}

template<> SEXP getHistory<double>(cppbugs::MCMCObject* node) {
  cppbugs::MCMCSpecialized<double>* sp = dynamic_cast<cppbugs::MCMCSpecialized<double>*>(node);
  if(sp == nullptr) {
    throw std::logic_error("invalid node conversion.");
  }
  //Rprintf("getHistory<double> history.size(): %d\n",sp->history.size());
  Rcpp::NumericVector ans(sp->history.size());
  R_len_t i = 0;
  for(typename std::list<double>::const_iterator it = sp->history.begin(); it != sp->history.end(); it++) {
    ans[i] =*it; ++i;
  }
  return Rcpp::wrap(ans);
}

SEXP makeNames(std::vector<const char*>& argnames) {
  SEXP ans;
  PROTECT(ans = Rf_allocVector(STRSXP, argnames.size()));
  for(size_t i = 0; i < argnames.size(); i++) {
    SET_STRING_ELT(ans, i, Rf_mkChar(argnames[i]));
  }
  UNPROTECT(1);
  return ans;
}

template<typename T>
void releaseMap(T& m) {
  for (typename T::iterator it=m.begin(); it != m.end(); it++) {
    delete it->second;
  }
}

SEXP createTrace(arglistT& arglist, vpArmaMapT& armaMap, vpMCMCMapT& mcmcMap) {
  SEXP ans; PROTECT(ans = Rf_allocVector(VECSXP, arglist.size()));
  for(size_t i = 0; i < arglist.size(); i++) {
    ArmaContext* ap = armaMap[rawAddress(arglist[i])];
    cppbugs::MCMCObject* node = mcmcMap[rawAddress(arglist[i])];
    if(!node->isObserved()) {
      switch(ap->getArmaType()) {
      case doubleT:
        SET_VECTOR_ELT(ans,i,getHistory<double>(node));
        break;
      case vecT:
        SET_VECTOR_ELT(ans,i,getHistory<arma::vec>(node));
        break;
      case matT:
      default:
        SET_VECTOR_ELT(ans,i,R_NilValue);
      }
    } else {
      SET_VECTOR_ELT(ans,i,R_NilValue);
    }
  }
  UNPROTECT(1);
  return ans;
}

SEXP runModel(SEXP m_, SEXP iterations, SEXP burn_in, SEXP adapt, SEXP thin) {
  SEXP env_ = Rf_getAttrib(m_,Rf_install("env"));
  if(env_ == R_NilValue || TYPEOF(env_) != ENVSXP) {
    throw std::logic_error("ERROR: bad environment passed to deterministic.");
  }

  vpArmaMapT armaMap;
  vpMCMCMapT mcmcMap;
  std::vector<cppbugs::MCMCObject*> mcmcObjects;

  arglistT arglist;
  std::vector<const char*> argnames;

  initArgList(m_, arglist, 1);
  for(size_t i = 0; i < arglist.size(); i++) {
    // force eval of late bindings
    if(TYPEOF(arglist[i])==SYMSXP) {
      argnames.push_back(CHAR(PRINTNAME(arglist[i])));
      arglist[i] = Rf_eval(arglist[i],env_);
    }
    ArmaContext* ap = getArma(arglist[i]);
    armaMap[rawAddress(arglist[i])] = ap;
    cppbugs::MCMCObject* node = createMCMC(arglist[i],armaMap);
    mcmcMap[rawAddress(arglist[i])] = node;
    mcmcObjects.push_back(node);
  }

  int iterations_ = Rcpp::as<int>(iterations);
  int burn_in_ = Rcpp::as<int>(burn_in);
  int adapt_ = Rcpp::as<int>(adapt);
  int thin_ = Rcpp::as<int>(thin);

  try {
    cppbugs::RMCModel m(mcmcObjects);
    m.sample(iterations_, burn_in_, adapt_, thin_);
    std::cout << "acceptance_ratio: " << m.acceptance_ratio() << std::endl;
  } catch (std::logic_error &e) {
    releaseMap(armaMap);
    releaseMap(mcmcMap);
    REprintf("%s\n",e.what());
    return R_NilValue;
  }

  SEXP ans;
  PROTECT(ans = createTrace(arglist,armaMap,mcmcMap));
  releaseMap(armaMap);
  releaseMap(mcmcMap);
  Rf_setAttrib(ans, R_NamesSymbol, makeNames(argnames));
  UNPROTECT(1);
  return ans;
}

ArmaContext* getArma(SEXP x_) {
  ArmaContext* ap;
  switch(TYPEOF(x_)) {
  case REALSXP:
    switch(getDims(x_).size()) {
    case 0: ap = new ArmaDouble(x_); break;
    case 1: ap = new ArmaVec(x_); break;
    case 2: ap = new ArmaMat(x_); break;
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
  return ap;
}

cppbugs::MCMCObject* createMCMC(SEXP x_, vpArmaMapT& armaMap) {
  SEXP distributed_sexp;
  distributed_sexp = Rf_getAttrib(x_,Rf_install("distributed"));
  if(distributed_sexp == R_NilValue) {
    throw std::logic_error("ERROR: 'distributed' attribute not defined. Is this an mcmc.object?");
  }

  if(armaMap.count(rawAddress(x_))==0) {
    throw std::logic_error("ArmaContext not found (object should be mapped before call to createMCMC).");
  }

  distT distributed = matchDistibution(std::string(CHAR(STRING_ELT(distributed_sexp,0))));
  cppbugs::MCMCObject* ans;

  switch(distributed) {
  case deterministicT:
    ans = createDeterministic(x_,armaMap);
    break;
  case normalDistT:
    ans = createNormal(x_,armaMap);
    break;
  case uniformDistT:
    ans = createUniform(x_,armaMap);
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

cppbugs::MCMCObject* createDeterministic(SEXP x_, vpArmaMapT& armaMap) {
  cppbugs::MCMCObject* p;
  ArmaContext* x_arma = armaMap[rawAddress(x_)];

  // function should be in position 1 (excluding fun/call name)
  SEXP fun_ = Rf_getAttrib(x_,Rf_install("update.method"));
  if(fun_ == R_NilValue || (TYPEOF(fun_) != CLOSXP && TYPEOF(fun_) != BCODESXP)) {
    throw std::logic_error("ERROR: update method must be a function.");
  }

  SEXP env_ = Rf_getAttrib(x_,Rf_install("env"));
  if(env_ == R_NilValue || TYPEOF(env_) != ENVSXP) {
    throw std::logic_error("ERROR: bad environment passed to deterministic.");
  }
  SEXP call_ = Rf_getAttrib(x_,Rf_install("call"));
  if(TYPEOF(call_) != LANGSXP) {
    throw std::logic_error("ERROR: function arguments not LANGSXP.");
  }
  if(Rf_length(call_) <= 2) {
    throw std::logic_error("ERROR: function must have at least one argument.");
  }

  arglistT arglist;
  initArgList(call_, arglist, 2);
  for(size_t i = 0; i < arglist.size(); i++) {
    if(TYPEOF(arglist[i])==SYMSXP) { arglist[i] = Rf_eval(arglist[i],env_); }
    getArma(arglist[i]); // for debug print
  }

  // map to arma types
  try {
    switch(x_arma->getArmaType()) {
    case doubleT:
      p = new cppbugs::RDeterministic<double>(x_arma->getDouble(),fun_,arglist);
      break;
    case vecT:
      p = new cppbugs::RDeterministic<arma::vec>(x_arma->getVec(),fun_,arglist);
      break;
    case matT:
      p = new cppbugs::RDeterministic<arma::mat>(x_arma->getMat(),fun_,arglist);
      break;
    case intT:
    case ivecT:
    case imatT:
    default:
      throw std::logic_error("ERROR: deterministic must be a continuous variable type (double, vec, or mat) for now (under development).");
    }
  } catch(std::logic_error &e) {
    REprintf("%s\n",e.what());
    return NULL;
  }
  return p;
}

cppbugs::MCMCObject* createNormal(SEXP x_,vpArmaMapT& armaMap) {
  cppbugs::MCMCObject* p;
  ArmaContext* x_arma = armaMap[rawAddress(x_)];

  SEXP env_ = Rf_getAttrib(x_,Rf_install("env"));
  SEXP mu_ = Rf_getAttrib(x_,Rf_install("mu"));
  SEXP tau_ = Rf_getAttrib(x_,Rf_install("tau"));
  SEXP observed_ = Rf_getAttrib(x_,Rf_install("observed"));

  //Rprintf("typeof mu: %d\n",TYPEOF(mu_));

  if(x_ == R_NilValue || env_ == R_NilValue || mu_ == R_NilValue || tau_ == R_NilValue || observed_ == R_NilValue) {
    REprintf("ERROR: missing argument.");
    return NULL;
  }

  // force substitutions
  if(TYPEOF(mu_)==SYMSXP) { mu_ = Rf_eval(mu_,env_); }
  if(TYPEOF(tau_)==SYMSXP) { tau_ = Rf_eval(tau_,env_); }

  bool observed = Rcpp::as<bool>(observed_);

  // map to arma types
  // these need to be in the armaMap to get cleaned up later
  ArmaContext* mu_arma = getArma(mu_);  armaMap[rawAddress(mu_)] = mu_arma;
  ArmaContext* tau_arma = getArma(tau_); armaMap[rawAddress(tau_)] = tau_arma;

  switch(x_arma->getArmaType()) {
  case doubleT:
    if(observed) {
      p = assignNormalLogp<cppbugs::ObservedNormal>(x_arma->getDouble(),mu_arma,tau_arma);
    } else {
      p = assignNormalLogp<cppbugs::Normal>(x_arma->getDouble(),mu_arma,tau_arma);
    }
    break;
  case vecT:
    if(observed) {
      p = assignNormalLogp<cppbugs::ObservedNormal>(x_arma->getVec(),mu_arma,tau_arma);
    } else {
      p = assignNormalLogp<cppbugs::Normal>(x_arma->getVec(),mu_arma,tau_arma);
    }
    break;
  case matT:
    if(observed) {
      p = assignNormalLogp<cppbugs::ObservedNormal>(x_arma->getMat(),mu_arma,tau_arma);
    } else {
      p = assignNormalLogp<cppbugs::Normal>(x_arma->getMat(),mu_arma,tau_arma);
    }
    break;
  case intT:
  case ivecT:
  case imatT:
  default:
    throw std::logic_error("ERROR: normal must be a continuous variable type (double, vec, or mat).");
  }
  return p;
}

cppbugs::MCMCObject* createUniform(SEXP x_,vpArmaMapT& armaMap) {
  cppbugs::MCMCObject* p;
  ArmaContext* x_arma = armaMap[rawAddress(x_)];

  SEXP env_ = Rf_getAttrib(x_,Rf_install("env"));
  SEXP lower_ = Rf_getAttrib(x_,Rf_install("lower"));
  SEXP upper_ = Rf_getAttrib(x_,Rf_install("upper"));
  SEXP observed_ = Rf_getAttrib(x_,Rf_install("observed"));

  if(x_ == R_NilValue || env_ == R_NilValue || lower_ == R_NilValue || upper_ == R_NilValue || observed_ == R_NilValue) {
    REprintf("ERROR: missing argument.");
    return NULL;
  }

  // force substitutions
  if(TYPEOF(lower_)==SYMSXP) { lower_ = Rf_eval(lower_,env_); }
  if(TYPEOF(upper_)==SYMSXP) { upper_ = Rf_eval(upper_,env_); }

  bool observed = Rcpp::as<bool>(observed_);

  // map to arma types
  // these need to be in the armaMap to get cleaned up later
  ArmaContext* lower_arma = getArma(lower_); armaMap[rawAddress(lower_)] = lower_arma;
  ArmaContext* upper_arma = getArma(upper_); armaMap[rawAddress(upper_)] = upper_arma;

  switch(x_arma->getArmaType()) {
  case doubleT:
    if(observed) {
      p = assignUniformLogp<cppbugs::ObservedUniform>(x_arma->getDouble(),lower_arma,upper_arma);
    } else {
      p = assignUniformLogp<cppbugs::Uniform>(x_arma->getDouble(),lower_arma,upper_arma);
    }
    break;
  case vecT:
    if(observed) {
      p = assignUniformLogp<cppbugs::ObservedUniform>(x_arma->getVec(),lower_arma,upper_arma);
    } else {
      p = assignUniformLogp<cppbugs::Uniform>(x_arma->getVec(),lower_arma,upper_arma);
    }
    break;
  case matT:
    if(observed) {
      p = assignUniformLogp<cppbugs::ObservedUniform>(x_arma->getMat(),lower_arma,upper_arma);
    } else {
      p = assignUniformLogp<cppbugs::Uniform>(x_arma->getMat(),lower_arma,upper_arma);
    }
    break;
  case intT:
  case ivecT:
  case imatT:
  default:
    throw std::logic_error("ERROR: uniform must be a continuous variable type (double, vec, or mat).");
  }
  return p;
}
