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

#ifndef ASSIGN_UNIFORM_LOGP_H
#define ASSIGN_UNIFORM_LOGP_H

#include "arma.context.h"
#include <cppbugs/distributions/mcmc.uniform.hpp>

template<template<typename> class MCTYPE, typename T>
MCTYPE<T>* assignUniformLogp(T& x, ArmaContext* lower, ArmaContext* upper) {
  MCTYPE<T>* node = new MCTYPE<T>(x);

  if(lower->getArmaType() == doubleT && upper->getArmaType() == doubleT) { node->dunif(lower->getDouble(),upper->getDouble()); }
  else if(lower->getArmaType() == vecT && upper->getArmaType() == doubleT) { node->dunif(lower->getVec(),upper->getDouble()); }
  else if(lower->getArmaType() == matT && upper->getArmaType() == doubleT) { node->dunif(lower->getMat(),upper->getDouble()); }
  else if(lower->getArmaType() == doubleT && upper->getArmaType() == vecT) { node->dunif(lower->getDouble(),upper->getVec()); }
  else if(lower->getArmaType() == vecT && upper->getArmaType() == vecT) { node->dunif(lower->getVec(),upper->getVec()); }
  else if(lower->getArmaType() == matT && upper->getArmaType() == vecT) { node->dunif(lower->getMat(),upper->getVec()); }
  else if(lower->getArmaType() == doubleT && upper->getArmaType() == matT) { node->dunif(lower->getDouble(),upper->getMat()); }
  else if(lower->getArmaType() == vecT && upper->getArmaType() == matT) { node->dunif(lower->getVec(),upper->getMat()); }
  else if(lower->getArmaType() == matT && upper->getArmaType() == matT) { node->dunif(lower->getMat(),upper->getMat()); }
  else { throw std::logic_error("ERROR: invalid type used in uniform distribution."); }

  return node;
}

#endif // ASSIGN_UNIFORM_LOGP_H
