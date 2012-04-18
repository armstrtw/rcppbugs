// -*- mode: C++; c-indent-level: 2; c-basic-offset: 2; tab-width: 8 -*-
///////////////////////////////////////////////////////////////////////////
// Copyright (C) 2012  Whit Armstrong                                    //
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

#ifndef ASSIGN_GAMMA_LOGP_H
#define ASSIGN_GAMMA_LOGP_H

#include "arma.context.h"
#include <cppbugs/mcmc.normal.hpp>

template<template<typename> class MCTYPE, typename T>
MCTYPE<T>* assignGammaLogp(T& x, ArmaContext* alpha, ArmaContext* beta) {
  MCTYPE<T>* node = new MCTYPE<T>(x);

  if(alpha->getArmaType() == doubleT && beta->getArmaType() == doubleT) { node->dgamma(alpha->getDouble(),beta->getDouble()); }
  else if(alpha->getArmaType() == vecT && beta->getArmaType() == doubleT) { node->dgamma(alpha->getVec(),beta->getDouble()); }
  else if(alpha->getArmaType() == matT && beta->getArmaType() == doubleT) { node->dgamma(alpha->getMat(),beta->getDouble()); }
  else if(alpha->getArmaType() == doubleT && beta->getArmaType() == vecT) { node->dgamma(alpha->getDouble(),beta->getVec()); }
  else if(alpha->getArmaType() == vecT && beta->getArmaType() == vecT) { node->dgamma(alpha->getVec(),beta->getVec()); }
  else if(alpha->getArmaType() == matT && beta->getArmaType() == vecT) { node->dgamma(alpha->getMat(),beta->getVec()); }
  else if(alpha->getArmaType() == doubleT && beta->getArmaType() == matT) { node->dgamma(alpha->getDouble(),beta->getMat()); }
  else if(alpha->getArmaType() == vecT && beta->getArmaType() == matT) { node->dgamma(alpha->getVec(),beta->getMat()); }
  else if(alpha->getArmaType() == matT && beta->getArmaType() == matT) { node->dgamma(alpha->getMat(),beta->getMat()); }
  else { throw std::logic_error("ERROR: invalid type used in normal distribution."); }

  return node;
}

#endif // ASSIGN_GAMMA_LOGP_H
