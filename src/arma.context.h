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
///////////////////////////////////////////////////////////////////////////

#ifndef ARMA_CONTEXT_H
#define ARMA_CONTEXT_H

#include <stdexcept>
#include <Rinternals.h>
#include <Rcpp.h>
#include <RcppArmadillo.h>

enum armaT { doubleT, vecT, matT, intT, ivecT, imatT };

class ArmaContext {
  armaT armatype_;
public:
  ArmaContext(armaT armatype): armatype_(armatype) {}
  const armaT getArmaType() const { return armatype_; }

  // double types
  virtual double& getDouble() { throw std::logic_error("ERROR: Arma type conversion not supported."); }
  virtual arma::vec& getVec() { throw std::logic_error("ERROR: Arma type conversion not supported."); }
  virtual arma::mat& getMat() { throw std::logic_error("ERROR: Arma type conversion not supported."); }

  // int types
  virtual int& getInt() { throw std::logic_error("ERROR: Arma type conversion not supported."); }
  virtual arma::ivec& getiVec() { throw std::logic_error("ERROR: Arma type conversion not supported."); }
  virtual arma::imat& getiMat() { throw std::logic_error("ERROR: Arma type conversion not supported."); }

  virtual void print() const = 0;
};

class ArmaDouble : public ArmaContext {
private:
  double& x_;
public:
  ArmaDouble(SEXP x_sexp): ArmaContext(doubleT), x_(REAL(x_sexp)[0]) {}
  double& getDouble() { return x_; }
  void print() const { std::cout << x_ << std::endl; }
};

class ArmaVec : public ArmaContext {
private:
  arma::vec x_;
public:
  ArmaVec(SEXP x_sexp): ArmaContext(vecT), x_(arma::vec(REAL(x_sexp), Rf_length(x_sexp), false)) {}
  arma::vec& getVec() { return x_; }
  void print() const { std::cout << x_ << std::endl; }
};

class ArmaMat : public ArmaContext {
private:
  arma::mat x_;
public:
  ArmaMat(SEXP x_sexp): ArmaContext(matT), x_(arma::mat(REAL(x_sexp), Rf_nrows(x_sexp), Rf_ncols(x_sexp), false)) {}
  arma::mat& getMat() { return x_; }
  void print() const { std::cout << x_ << std::endl; }
};


class ArmaInt : public ArmaContext {
private:
  int& x_;
public:
  ArmaInt(SEXP x_sexp): ArmaContext(intT), x_(INTEGER(x_sexp)[0]) {}
  int& getInt() { return x_; }
  void print() const { std::cout << x_ << std::endl; }
};

class ArmaiVec : public ArmaContext {
private:
  arma::ivec x_;
public:
  ArmaiVec(SEXP x_sexp): ArmaContext(ivecT), x_(arma::ivec(INTEGER(x_sexp), Rf_length(x_sexp), false)) {}
  arma::ivec& getiVec() { return x_; }
  void print() const { std::cout << x_ << std::endl; }
};

class ArmaiMat : public ArmaContext {
private:
  arma::imat x_;
public:
  ArmaiMat(SEXP x_sexp): ArmaContext(imatT), x_(arma::imat(INTEGER(x_sexp), Rf_nrows(x_sexp), Rf_ncols(x_sexp), false)) {}
  arma::imat& getiMat() { return x_; }
  void print() const { std::cout << x_ << std::endl; }
};


#endif // ARMA_CONTEXT_H
