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

#ifndef ASSIGN_BINOMIAL_LOGP_H
#define ASSIGN_BINOMIAL_LOGP_H

#include "arma.context.h"
#include <cppbugs/mcmc.binomial.hpp>

template<template<typename> class MCTYPE, typename T>
MCTYPE<T>* assignBinomialLogp(T& x, ArmaContext* n, ArmaContext* p) {
  MCTYPE<T>* node = new MCTYPE<T>(x);

  if(n->getArmaType() == intT && p->getArmaType() == doubleT) { node->dbinom(n->getInt(),p->getDouble()); }
  else if(n->getArmaType() == ivecT && p->getArmaType() == doubleT) { node->dbinom(n->getiVec(),p->getDouble()); }
  else if(n->getArmaType() == imatT && p->getArmaType() == doubleT) { node->dbinom(n->getiMat(),p->getDouble()); }
  else if(n->getArmaType() == intT && p->getArmaType() == vecT) { node->dbinom(n->getInt(),p->getVec()); }
  else if(n->getArmaType() == ivecT && p->getArmaType() == vecT) { node->dbinom(n->getiVec(),p->getVec()); }
  else if(n->getArmaType() == imatT && p->getArmaType() == vecT) { node->dbinom(n->getiMat(),p->getVec()); }
  else if(n->getArmaType() == intT && p->getArmaType() == matT) { node->dbinom(n->getInt(),p->getMat()); }
  else if(n->getArmaType() == ivecT && p->getArmaType() == matT) { node->dbinom(n->getiVec(),p->getMat()); }
  else if(n->getArmaType() == imatT && p->getArmaType() == matT) { node->dbinom(n->getiMat(),p->getMat()); }
  else { throw std::logic_error("ERROR: invalid type used in binomial distribution."); }

  return node;
}

#endif // ASSIGN_BINOMIAL_LOGP_H
