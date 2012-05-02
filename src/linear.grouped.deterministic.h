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

#ifndef LINEAR_GROUPED_DETERMINISTIC_H
#define LINEAR_GROUPED_DETERMINISTIC_H

#include <RcppArmadillo.h>
#include <cppbugs/mcmc.deterministic.hpp>


namespace cppbugs {

  template<typename T>
  class LinearGroupedDeterministic : public Deterministic<arma::mat> {
  private:
    const T& X_;
    const arma::mat& b_;
    const arma::ivec& group_;
    arma::uvec group_0_index_;
  public:
    LinearGroupedDeterministic(arma::mat& value, const T& X, const arma::mat& b, const arma::ivec group):
      Deterministic<arma::mat>(value), X_(X), b_(b), group_(group), group_0_index_(arma::conv_to<arma::uvec>::from(group - 1))
    {
      int max_index = max(group);
      int min_index = min(group);
      if(max_index > b.n_rows || min_index < 1) {
        throw std::logic_error("ERROR: createLinearGroupedDeterministic, group index is out of range of rows of matrix b.");
      }
      if(group.n_elem != X.n_rows) {
        throw std::logic_error("ERROR: createLinearGroupedDeterministic, group index does not match number of rows of X.");
      }
    }

    void jump(RngBase& rng) {
      //group_0_index = group - 1;
      Deterministic<arma::mat>::value = arma::sum(X_ % b_.rows(group_0_index_),1);
      //Deterministic<arma::mat>::value = arma::sum(X_ %*% b_.rows(group_-1),1);
    }
  };
} // namespace cppbugs
#endif //LINEAR_GROUPED_DETERMINISTIC_H
