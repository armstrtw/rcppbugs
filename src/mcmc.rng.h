///////////////////////////////////////////////////////////////////////////
// Copyright (C) 2011 Whit Armstrong                                     //
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

#ifndef MCMC_R_NATIVE_RNG_H
#define MCMC_R_NATIVE_RNG_H

#include <cppbugs/mcmc.rng.base.hpp>
#include <S.h>

namespace cppbugs {

  class RNativeRng : public RngBase {
  public:
    RNativeRng() { GetRNGstate(); }
    ~RNativeRng() { PutRNGstate(); }
    double normal() { return norm_rand(); }
    double uniform() { return unif_rand(); }
  };

} // namespace cppbugs
#endif // MCMC_R_NATIVE_RNG_H
