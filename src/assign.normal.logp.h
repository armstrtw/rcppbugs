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

#ifndef ASSIGN_NORMAL_LOGP_H
#define ASSIGN_NORMAL_LOGP_H

#include <cppbugs/cppbugs.hpp>
#include "arma.context.h"

#ifdef dnorm
#undef dnorm

template<template<typename> class MCTYPE, typename T>
MCTYPE<T>* assignNormalLogp(T& x, ArmaContext* mu, ArmaContext* tau) {
  MCTYPE<T>* node = new MCTYPE<T>(x);

  if(mu->getArmaType() == doubleT && tau->getArmaType() == doubleT) { node->dnorm(mu->getDouble(),tau->getDouble()); }
  else if(mu->getArmaType() == vecT && tau->getArmaType() == doubleT) { node->dnorm(mu->getVec(),tau->getDouble()); }
  else if(mu->getArmaType() == matT && tau->getArmaType() == doubleT) { node->dnorm(mu->getMat(),tau->getDouble()); }
  else if(mu->getArmaType() == doubleT && tau->getArmaType() == vecT) { node->dnorm(mu->getDouble(),tau->getVec()); }
  else if(mu->getArmaType() == vecT && tau->getArmaType() == vecT) { node->dnorm(mu->getVec(),tau->getVec()); }
  else if(mu->getArmaType() == matT && tau->getArmaType() == vecT) { node->dnorm(mu->getMat(),tau->getVec()); }
  else if(mu->getArmaType() == doubleT && tau->getArmaType() == matT) { node->dnorm(mu->getDouble(),tau->getMat()); }
  else if(mu->getArmaType() == vecT && tau->getArmaType() == matT) { node->dnorm(mu->getVec(),tau->getMat()); }
  else if(mu->getArmaType() == matT && tau->getArmaType() == matT) { node->dnorm(mu->getMat(),tau->getMat()); }
  else { throw std::logic_error("ERROR: invalid type used in normal distribution."); }

  return node;
}

#endif // #undef dnorm

#endif // ASSIGN_NORMAL_LOGP_H
