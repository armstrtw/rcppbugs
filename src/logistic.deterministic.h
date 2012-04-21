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

#ifndef LOGISTIC_DETERMINISTIC_H
#define LOGISTIC_DETERMINISTIC_H

//#include <Rcpp.h>
#include <RcppArmadillo.h>
#include <cppbugs/mcmc.deterministic.hpp>


namespace cppbugs {

  template<typename T>
  class LogisticDeterministic : public Deterministic<arma::mat> {
  private:
    const T& X_;
    const arma::vec& b_;
  public:
    LogisticDeterministic(arma::mat& value, const T& X, const arma::vec& b):
      Deterministic<arma::mat>(value), X_(X), b_(b) {}

    void jump(RngBase& rng) {      
      Deterministic<arma::mat>::value = 1/(1+exp(-X_*b_));
    }
  };
} // namespace cppbugs
#endif //LOGISTIC_DETERMINISTIC_H
