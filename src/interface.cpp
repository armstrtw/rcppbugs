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
//#include "interface.h"
#include "helpers.h"
#include "finalizers.h"
#include "raw.address.h"
#include "distribution.types.h"
#include "arma.context.h"
#include "assign.normal.logp.h"
#include "assign.uniform.logp.h"
#include "r.deterministic.h"
#include "r.mcmc.model.h"

// public interface
extern "C" SEXP getRawAddr(SEXP x);
extern "C" SEXP createRef(SEXP x_);
extern "C" SEXP getRef(SEXP x_);
extern "C" SEXP modmem(SEXP x_);
extern "C" SEXP logp(SEXP x);
extern "C" SEXP jump(SEXP x);
extern "C" SEXP printArma(SEXP x);
extern "C" SEXP printMCMC(SEXP x);
extern "C" SEXP createModel(SEXP args_sexp);
extern "C" SEXP run_model(SEXP mp_, SEXP iterations, SEXP burn_in, SEXP adapt, SEXP thin);
extern "C" SEXP createDeterministic(SEXP args_);
extern "C" SEXP createNormal(SEXP x_, SEXP mu_, SEXP tau_, SEXP observed_);
extern "C" SEXP createUniform(SEXP x_, SEXP lower_, SEXP upper_, SEXP observed_);
extern "C" SEXP getHist(SEXP x);
extern "C" SEXP attachArgs(SEXP args);

// private methods
void* getExternalPointer(SEXP p_);
void* getExternalPointerAttribute(SEXP x_, const char* attyName);
ArmaContext* getArma(SEXP x);
cppbugs::MCMCObject* getMCMC(SEXP x_);
void setMCMC(SEXP x_, cppbugs::MCMCObject* p);
SEXP createRef(SEXP x_);
void addRef(SEXP x_, SEXP ref, const char* refname);

class ObjectRef {
private:
  SEXPREC& sexp_;
public:
  ObjectRef(SEXP sexp): sexp_(*sexp) {}
  SEXP getSEXP() const { return &sexp_; }
};


SEXP getRawAddr(SEXP x) {
  if(x != R_NilValue) {
    Rprintf("sexp: %p raw: %p\n",x,rawAddress(x));
  } else {
    Rprintf("<nil>\n");
  }
  return R_NilValue;
}

SEXP modmem(SEXP x_) {
  REAL(x_)[0] = 100;
}

static void objectRefFinalizer(SEXP o_) {
  finalizeSEXP<ObjectRef>(o_);
}

/*
static void armaContextFinalizer(SEXP a_) {
  finalizeSEXP<ArmaContext>(a_);
}
*/

static void modelFinalizer(SEXP m_) {
  finalizeSEXP<cppbugs::MCModel<boost::minstd_rand> >(m_);
}


static void arglistFinalizer(SEXP a_) {
  finalizeSEXP<arglistT>(a_);
}

static void mcmcObjectFinalizer(SEXP o_) {
  finalizeSEXP<cppbugs::MCMCObject>(o_);
}

SEXP createRef(SEXP x_) {
  ObjectRef* p = new ObjectRef(x_);
  return createExternalPoniter(p, objectRefFinalizer, "ObjectRef*");
}

SEXP getRef(SEXP x_) {
  ObjectRef* p = reinterpret_cast<ObjectRef*>(getExternalPointer(x_));
  return p->getSEXP();
}

void addRef(SEXP x_, SEXP ref, const char* refname) {
  SEXP ref_;
  PROTECT(ref_ = createRef(ref));
  Rf_setAttrib(x_, Rf_install(refname), ref_);
  UNPROTECT(1);
}

void* getExternalPointer(SEXP p_) {
  if(p_ == R_NilValue || TYPEOF(p_) != EXTPTRSXP) {
    throw std::logic_error("ERROR: invalid external pointer SEXP object.\n");
  }
  void* p = R_ExternalPtrAddr(p_);
  if(!p) {
    throw std::logic_error("ERROR: bad pointer conversion.\n");
  }
  return p;
}

void* getExternalPointerAttribute(SEXP x_, const char* attyName) {
  SEXP p_ = Rf_getAttrib(x_, Rf_install(attyName));
  return getExternalPointer(p_);
}


SEXP attachArgs(SEXP args) {
  args = CDR(args); // skip 'name'

  // pull off data object
  SEXP x = CAR(args); args = CDR(args);
  arglistT* arglist = new arglistT;

  // loop through rest of args
  for(; args != R_NilValue; args = CDR(args)) {
    arglist->push_back(CAR(args));
  }

  
  Rf_setAttrib(x, Rf_install("args"), createExternalPoniter(p, arglistFinalizer, "arglist*"));
  return R_NilValue;
}


SEXP logp(SEXP x) {
  double ans = std::numeric_limits<double>::quiet_NaN();
  cppbugs::MCMCObject* node(NULL);
  try {
    node = getMCMC(x);
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

SEXP jump(SEXP x) {
  static cppbugs::SpecializedRng<boost::minstd_rand> rng;

  cppbugs::MCMCObject* node(NULL);
  try {
    node = getMCMC(x);
  } catch (std::logic_error &e) {
    REprintf("%s\n",e.what());
    return R_NilValue;
  }
  node->jump(rng);
  return R_NilValue;
}

SEXP printMCMC(SEXP x) {
  cppbugs::MCMCObject* node(NULL);
  try {
    node = getMCMC(x);
    node->print();
  } catch (std::logic_error &e) {
    REprintf("%s\n",e.what());
    return R_NilValue;
  }
  return R_NilValue;
}

SEXP printArma(SEXP x) {
  ArmaContext* node(NULL);
  try {
    node = getArma(x);
    node->print();
  } catch (std::logic_error &e) {
    REprintf("%s\n",e.what());
    return R_NilValue;
  }
  return R_NilValue;
}

SEXP createModel(SEXP args_sexp) {
  std::vector<cppbugs::MCMCObject*> mcmcObjects;
  try {
    //CDR(args_sexp); // function name
    //for(R_len_t i = 0; i < Rf_length(args_sexp); i++) {
    args_sexp = CDR(args_sexp); /* skip 'name' */
    for(int i = 0; args_sexp != R_NilValue; i++, args_sexp = CDR(args_sexp)) {
      SEXP this_sexp = CAR(args_sexp);
      mcmcObjects.push_back(getMCMC(this_sexp));      
    }
  } catch (std::logic_error &e) {
    REprintf("%s\n",e.what());
    return R_NilValue;
  }
  cppbugs::MCModel<boost::minstd_rand>* m = new cppbugs::MCModel<boost::minstd_rand>(mcmcObjects);
  return createExternalPoniter(m, modelFinalizer, "cppbugs::MCModel<boost::minstd_rand>*");
}

SEXP run_model(SEXP m_, SEXP iterations, SEXP burn_in, SEXP adapt, SEXP thin) {

  int iterations_ = Rcpp::as<int>(iterations);
  int burn_in_ = Rcpp::as<int>(burn_in);
  int adapt_ = Rcpp::as<int>(adapt);
  int thin_ = Rcpp::as<int>(thin);

  try {
    cppbugs::MCModel<boost::minstd_rand>* m = reinterpret_cast<cppbugs::MCModel<boost::minstd_rand>* >(getExternalPointer(m_));
    m->sample(iterations_, burn_in_, adapt_, thin_);
    return Rcpp::wrap(m->acceptance_ratio());
  } catch (std::logic_error &e) {
    REprintf("%s\n",e.what());
    return R_NilValue;
  }
}

/*
void addArmaContext(SEXP x, ArmaContext* ap) {
  PROTECT(ap_ = R_MakeExternalPtr(reinterpret_cast<void*>(ap),Rf_install("ArmaContext*"),R_NilValue));
  R_RegisterCFinalizerEx(ap_, armaContextFinalizer, TRUE);
  Rf_setAttrib(x, Rf_install("armaContext"), ap_);
  UNPROTECT(1);
}
*/

// adds an armaContext external pointer if it does not exist
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

// unlike getArma, this throws if MCMC external pointer is not found
cppbugs::MCMCObject* getMCMC(SEXP x_) {
  cppbugs::MCMCObject* p(NULL);
  try {
    p = reinterpret_cast<cppbugs::MCMCObject*>(getExternalPointerAttribute(x_, "mcmc_ptr"));
  } catch(std::logic_error &e) {
    REprintf("%s\n",e.what());
  }
  return p;
}

void setMCMC(SEXP x_, cppbugs::MCMCObject* p) {
  Rf_setAttrib(x_, Rf_install("mcmc_ptr"), createExternalPoniter(p, mcmcObjectFinalizer, "cppbugs::MCMCObject*"));
}

cppbugs::MCMCObject* createDeterministic(SEXP x_) {
  cppbugs::MCMCObject* p;
  Rprintf("deterministic sexp: %p raw: %p\n",x_,rawAddress(x_));

  // function should be in position 1 (excluding fun/call name)
  SEXP fun_ = Rf_getAttrib(x_,install("update.method"));

  if(fun_ == R_NilValue || TYPEOF(fun_) != CLOSXP) {
    throw std::logic_error("ERROR: update method must be a function.");
  }

  arglistT arglist = *reinterpret_cast<arglistT*>(getExternalPointerAttribute(x,"args"));

  // map to arma types
  ArmaContext* x_arma = getArma(x_);
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
    return R_NilValue;
  }
  return p;
}

cppbugs::MCMCObject* createNormal(SEXP x_, SEXP mu_, SEXP tau_, SEXP observed_) {
  cppbugs::MCMCObject* p;
  Rprintf("normal sexp: %p raw: %p\n",x_,rawAddress(x_));

  if(mu_ == R_NilValue || tau_ == R_NilValue || observed_ == R_NilValue) {
    REprintf("ERROR: missing argument.");
    return R_NilValue;
  }

  bool observed = Rcpp::as<bool>(observed_);

  // map to arma types
  ArmaContext* x_arma = getArma(x_); Rprintf("x addr: %p\n",rawAddress(x_));
  ArmaContext* mu_arma = getArma(mu_);  Rprintf("mu addr: %p\n",rawAddress(mu_));
  ArmaContext* tau_arma = getArma(tau_); Rprintf("tau addr: %p\n",rawAddress(tau_));

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

cppbugs::MCMCObject* createUniform(SEXP x_, SEXP lower_, SEXP upper_, SEXP observed_) {
   cppbugs::MCMCObject* p;
   Rprintf("uniform sexp: %p raw: %p\n",x_,rawAddress(x_));

  if(lower_ == R_NilValue || upper_ == R_NilValue || observed_ == R_NilValue) {
    REprintf("ERROR: missing argument.");
    return R_NilValue;
  }

  bool observed = Rcpp::as<bool>(observed_);

  // map to arma types
  // FIXME: delete later...
  ArmaContext* x_arma = getArma(x_);
  ArmaContext* lower_arma = getArma(lower_);
  ArmaContext* upper_arma = getArma(upper_);

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


template<typename T>
SEXP getHistory(cppbugs::MCMCObject* node) {
  //SEXP ans;
  cppbugs::MCMCSpecialized<T>* sp = dynamic_cast<cppbugs::MCMCSpecialized<T>*>(node);
  if(sp == nullptr) {
    throw std::logic_error("invalid node conversion.");
  }
  Rprintf("getHistory<T> history.size(): %d\n",sp->history.size());
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
  Rprintf("getHistory<arma::vec> history.size(): %d\n",sp->history.size());
  const size_t NC = sp->history.begin()->n_elem;
  Rprintf("getHistory<arma::vec> history dim: %d\n",NC);
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
  Rprintf("getHistory<double> history.size(): %d\n",sp->history.size());
  Rcpp::NumericVector ans(sp->history.size());
  R_len_t i = 0;
  for(typename std::list<double>::const_iterator it = sp->history.begin(); it != sp->history.end(); it++) {
    ans[i] =*it; ++i;
  }
  return Rcpp::wrap(ans);
}


SEXP getHist(SEXP x_) {
  SEXP ans;
  ArmaContext* ap(NULL);
  cppbugs::MCMCObject* node(NULL);
  try {
    ap = getArma(x_);
    node = getMCMC(x_);
  } catch (std::logic_error &e) {
    REprintf("%s\n",e.what());
    return R_NilValue;
  }

  try {
    switch(ap->getArmaType()) {
    case doubleT:
      PROTECT(ans = getHistory<double>(node));
      break;
    case vecT:
      PROTECT(ans = getHistory<arma::vec>(node));
      break;
    case matT:
      PROTECT(ans = getHistory<arma::mat>(node));
      break;
    case intT:
    case ivecT:
    case imatT:
    default:
      throw std::logic_error("ERROR: history conversion not supported for this type.");
    }
  } catch (std::logic_error &e) {
    REprintf("%s\n",e.what());
    return R_NilValue;
  }
  UNPROTECT(1);
  return Rcpp::wrap(ans);
}

/*
void appendHistory(sexpMCMCMapT& mcmc_map, sexpArmaMapT& arma_map, SEXP x) {
  SEXP ans;
  void* sexp_data_ptr = rawAddress(x);

  if(arma_map.count(sexp_data_ptr) == 0) {
    throw std::logic_error("ERROR: getHistory, node not found in armaContextMap.");
  }

  if(mcmc_map.count(sexp_data_ptr) == 0) {
    throw std::logic_error("ERROR: getHistory, node not found in mcmcContextMap.");
  }

  ArmaContext* arma_ptr = arma_map[sexp_data_ptr];
  cppbugs::MCMCObject* mcmc_ptr = mcmc_map[sexp_data_ptr];

  switch(arma_ptr->getArmaType()) {
  case doubleT:
    PROTECT(ans = getHistory<double>(mcmc_ptr));
    break;
  case vecT:
    PROTECT(ans = getHistory<arma::vec>(mcmc_ptr));
    break;
  case matT:
    PROTECT(ans = getHistory<arma::mat>(mcmc_ptr));
    break;
  case intT:
  case ivecT:
  case imatT:
  default:
    throw std::logic_error("ERROR: history conversion not supported for this type.");
  }
  //Rprintf("returned length: %d",Rf_length(ans));
  Rf_setAttrib(x, Rf_install("history"), ans);
  UNPROTECT(1);
}
*/
