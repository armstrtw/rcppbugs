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
#include <iostream>
#include <Rinternals.h>
#include <cppbugs/mcmc.dynamic.hpp>

typedef std::vector<SEXP> arglistT;

namespace cppbugs {

  template<typename T>
  class RDeterministic : public Dynamic<T> {
    SEXP call_, rho_;

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
      //std::cout << "address of updated mat:" << dest.memptr() << std::endl;
    }
    
  public:
    void jump(RngBase& rng) {
      SEXP ans;
      PROTECT(ans = Rf_eval(call_,rho_));
      updateFromSEXP(Dynamic<T>::value,ans);
      UNPROTECT(1);
      //std::cout << "RDeterministic new value:"  << std::endl << Dynamic<T>::value << std::endl;
    }
    ~RDeterministic() { UNPROTECT(1); } // for call_
    RDeterministic(T& value, SEXP fun, SEXP args, SEXP rho): Dynamic<T>(value), rho_(rho) {
      PROTECT(call_ = Rf_lcons(fun,args));
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
    double getScale() const { return 0; }
  };

} // namespace cppbugs
#endif //R_DETERMINISTIC_H
