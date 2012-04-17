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
  virtual double& getDouble() { throw std::logic_error("ERROR: Arma type conversion not supported."); }
  virtual arma::vec& getVec() { throw std::logic_error("ERROR: Arma type conversion not supported."); }
  virtual arma::mat& getMat() { throw std::logic_error("ERROR: Arma type conversion not supported."); }
  virtual void print() const = 0;

  // virtual double& getDouble() const { throw std::logic_error("ERROR: Arma type conversion not supported."); double bad; return bad; }
  // virtual arma::vec& getVec() const { throw std::logic_error("ERROR: Arma type conversion not supported."); arma::vec bad; return bad; }
  // virtual arma::mat& getMat() const { throw std::logic_error("ERROR: Arma type conversion not supported."); arma::mat bad; return bad; }
  // virtual int& getInt() const { throw std::logic_error("ERROR: Arma type conversion not supported."); }
  // virtual ivec& getIVec() const { throw std::logic_error("ERROR: Arma type conversion not supported."); }
  // virtual imat& getIMat() const { throw std::logic_error("ERROR: Arma type conversion not supported."); }
};

class ArmaDouble : public ArmaContext {
private:
  double& x_;
public:
  ArmaDouble(SEXP x_sexp): ArmaContext(doubleT), x_(REAL(x_sexp)[0]) {
    //std::cout << "double, raw mem: " << REAL(x_sexp) << std::endl;
  }
  double& getDouble() { return x_; }
  void print() const { std::cout << x_ << std::endl; }
};

class ArmaVec : public ArmaContext {
private:
  arma::vec x_;
public:
  //ArmaVec(SEXP x_sexp): ArmaContext(vecT), x_(Rcpp::as<arma::vec>(x_sexp)) {}
  ArmaVec(SEXP x_sexp): ArmaContext(vecT), x_(arma::vec(REAL(x_sexp), Rf_length(x_sexp), false)) {
    //std::cout << "vec, raw mem: " << REAL(x_sexp) << std::endl;
  }
  arma::vec& getVec() { return x_; }
  void print() const { std::cout << x_ << std::endl; }
};

class ArmaMat : public ArmaContext {
private:
  arma::mat x_;
public:
  //ArmaMat(SEXP x_sexp): ArmaContext(matT), x_(Rcpp::as<arma::mat>(x_sexp)) {}
  ArmaMat(SEXP x_sexp): ArmaContext(matT), x_(arma::mat(REAL(x_sexp), Rf_nrows(x_sexp), Rf_ncols(x_sexp), false)) {
    //std::cout << "mat, raw mem: " << REAL(x_sexp) << std::endl;
  }
  arma::mat& getMat() { return x_; }
  void print() const { std::cout << x_ << std::endl; }
};


#endif // ARMA_CONTEXT_H
