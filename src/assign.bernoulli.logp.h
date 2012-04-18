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

#ifndef ASSIGN_BERNOULLI_LOGP_H
#define ASSIGN_BERNOULLI_LOGP_H

#include "arma.context.h"
#include <cppbugs/mcmc.bernoulli.hpp>

template<template<typename> class MCTYPE, typename T>
MCTYPE<T>* assignBernoulliLogp(T& x, ArmaContext* p) {
  MCTYPE<T>* node = new MCTYPE<T>(x);

  if(p->getArmaType() == doubleT) { node->dbern(p->getDouble()); }
  else if(p->getArmaType() == vecT) { node->dbern(p->getVec()); }
  else if(p->getArmaType() == matT) { node->dbern(p->getMat()); }
  else { throw std::logic_error("ERROR: invalid type used in bernoulli distribution."); }

  return node;
}

#endif // ASSIGN_BERNOULLI_LOGP_H
