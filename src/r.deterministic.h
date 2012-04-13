// -*- mode: C++; c-indent-level: 2; c-basic-offset: 2; tab-width: 8 -*-
///////////////////////////////////////////////////////////////////////////
// Copyright (C) 2012 Whit Armstrong                                     //
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

#ifndef R_DETERMINISTIC_H
#define R_DETERMINISTIC_H

//#include <R.h>
//#include <Rdefines.h>
#include <Rinternals.h>
#include <cppbugs/mcmc.dynamic.hpp>


/*
double* memptr(double& x) {
  return &x;
}

int* memptr(int& x) {
  return &x;
}

double* memptr(vec& x) {
  return x.memptr();
}

double* memptr(mat& x) {
  return x.memptr();
}

int* memptr(ivec& x) {
  return x.memptr();
}

int* memptr(imat& x) {
  return x.memptr();
}
*/

namespace cppbugs {

  template<typename T>
  class RDeterministic : public Dynamic<T> {
    SEXP fo_;

    static void updateFromSEXP(double& dest, SEXP x) {
      dest = REAL(x)[0];
    }

    static void updateFromSEXP(int& dest, SEXP x) {
      dest = INTEGER(x)[0];
    }

    static void updateFromSEXP(arma::vec& dest, SEXP x) {
      //Rprintf("dest size: %d\n",dest.n_elem);
      //Rprintf("src sizee: %d\n", Rf_length(x));
      memcpy(dest.memptr(),REAL(x),sizeof(double)*dest.n_elem);
    }

    static void updateFromSEXP(arma::mat& dest, SEXP x) {
      //Rprintf("dest size: %d\n",dest.n_elem);
      memcpy(dest.memptr(),REAL(x),sizeof(double)*dest.n_elem);
    }

  public:
    void jump(RngBase& rng) {
      SEXP ans;
      //Dynamic<T>::value = as<typename T>::(eval(fo_, env_));
      PROTECT(ans = Rf_eval(fo_, R_GlobalEnv));
      //Rprintf("RDeterministic size: %d\n", Rf_length(ans));
      updateFromSEXP(Dynamic<T>::value,ans);
      UNPROTECT(1);
    }
    ~RDeterministic() { UNPROTECT(1); }

    RDeterministic(T& value, SEXP fun, SEXP arg0): Dynamic<T>(value) {
      PROTECT(fo_ = Rf_lang2(fun, arg0));
    }

    RDeterministic(T& value, SEXP fun, SEXP arg0, SEXP arg1): Dynamic<T>(value) {
      PROTECT(fo_ = Rf_lang3(fun, arg0, arg1));
    }

    RDeterministic(T& value, SEXP fun, SEXP arg0, SEXP arg1, SEXP arg2): Dynamic<T>(value) {
      PROTECT(fo_ = Rf_lang4(fun, arg0, arg1, arg2));
    }

    RDeterministic(T& value, SEXP fun, SEXP arg0, SEXP arg1, SEXP arg2, SEXP arg3): Dynamic<T>(value) {
      PROTECT(fo_ = Rf_lang5(fun, arg0, arg1, arg2, arg3));
    }

    void accept() {}
    void reject(){}
    void tune() {}
    // in Dynamic: void preserve()
    // in Dynamic: void revert()
    // in Dynamic: void tally()
    bool isDeterministc() const { return true; }
    bool isStochastic() const { return false; }
    bool isObserved() const { return false; }
    void setScale(const double scale) {}
  };

} // namespace cppbugs
#endif //R_DETERMINISTIC_H
