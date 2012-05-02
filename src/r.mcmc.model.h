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

#ifndef R_MCMC_MODEL_H
#define R_MCMC_MODEL_H

#include <iostream>
#include <cmath>
#include <vector>
#include <map>
#include <exception>
#include <cppbugs/mcmc.object.hpp>
#include <cppbugs/mcmc.stochastic.hpp>
#include "mcmc.rng.h"

namespace cppbugs {

  class RMCModel {
  private:
    double accepted_,rejected_,logp_value_,old_logp_value_;
    RNativeRng rng_;
    std::vector<MCMCObject*> mcmcObjects_, dynamic_nodes, determinsitic_nodes;
    std::vector<Likelihiood*> logp_functors;
   
    void jump() { for(size_t i = 0; i < dynamic_nodes.size(); i++) { dynamic_nodes[i]->jump(rng_); } }
    void jump_detrministics() { for(size_t i = 0; i < determinsitic_nodes.size(); i++) { determinsitic_nodes[i]->jump(rng_); } }
    void preserve() { for(size_t i = 0; i < dynamic_nodes.size(); i++) { dynamic_nodes[i]->preserve(); } }
    void revert() { for(size_t i = 0; i < dynamic_nodes.size(); i++) { dynamic_nodes[i]->revert(); } }
    void set_scale(const double scale) { for(size_t i = 0; i < dynamic_nodes.size(); i++) { dynamic_nodes[i]->setScale(scale); } }
    void tally() { for(size_t i = 0; i < dynamic_nodes.size(); i++) { dynamic_nodes[i]->tally(); } }
    //void print() { for(auto v : mcmcObjects_) { v->print(); } }
    static bool bad_logp(const double value) { return std::isnan(value) || value == -std::numeric_limits<double>::infinity() ? true : false; }

    void addStochcasticNode(MCMCObject* node) {
      Stochastic* sp = dynamic_cast<Stochastic*>(node);
      // FIXME: this should throw if sp->getLikelihoodFunctor() returns null
      if(sp && sp->getLikelihoodFunctor() ) {
        //std::cout << "adding stochastic" << std::endl;
        logp_functors.push_back(sp->getLikelihoodFunctor());
      }
    }

    void initChain() {
      for(std::vector<MCMCObject*>::iterator node = mcmcObjects_.begin(); node != mcmcObjects_.end(); node++) {
        // FIXME: add test here to check starting from invalid logp or NaN
        addStochcasticNode(*node);

        if((*node)->isDeterministc()) {
          determinsitic_nodes.push_back(*node);
        }

        if(!(*node)->isObserved()) {
          dynamic_nodes.push_back(*node);
        }
      }
      // init logp
      logp_value_ = logp();
    }

    void resetAcceptanceRatio() {
      accepted_ = 0;
      rejected_ = 0;
    }

    bool reject(const double value, const double old_logp) {
      return bad_logp(value) || log(rng_.uniform()) > (value - old_logp) ? true : false;
    }

    void tune(int iterations, int tuning_step) {
      for(int i = 1; i <= iterations; i++) {
	for(std::vector<MCMCObject*>::iterator it = dynamic_nodes.begin(); it != dynamic_nodes.end(); it++) {
          old_logp_value_ = logp_value_;
          (*it)->preserve();
          (*it)->jump(rng_);

          // has to be done after each stoch jump
          jump_detrministics();
          logp_value_ = logp();
          if(reject(logp_value_, old_logp_value_)) {
            (*it)->revert();
            logp_value_ = old_logp_value_;
            (*it)->reject();
          } else {
            (*it)->accept();
          }
	}
	if(i % tuning_step == 0) {
	  for(std::vector<MCMCObject*>::iterator itdyn = dynamic_nodes.begin(); itdyn != dynamic_nodes.end(); itdyn++) {
	    (*itdyn)->tune();
	  }
	}
      }
    }

    void step() {
      old_logp_value_ = logp_value_;
      preserve();
      jump();
      logp_value_ = logp();
      if(reject(logp_value_, old_logp_value_)) {
        revert();
        logp_value_ = old_logp_value_;
        rejected_ += 1;
      } else {
        accepted_ += 1;
      }
    }

    void tune_global(int iterations, int tuning_step) {
      const double thresh = 0.1;
      // FIXME: this should possibly related to the overall size/dimension
      // of the parmaeters to be estimtated, as there is somewhat of a leverage effect
      // via the number of parameters
      const double dilution = 0.10;
      double total_size = 0;

      for(size_t i = 0; i < dynamic_nodes.size(); i++) {
        if(dynamic_nodes[i]->isStochastic()) {
          total_size += dynamic_nodes[i]->size();
        }
      }
      double target_ar = std::max(1/log2(total_size + 3), 0.234);
      for(int i = 1; i <= iterations; i++) {
        step();
        if(i % tuning_step == 0) {
          double diff = acceptance_ratio() - target_ar;
          resetAcceptanceRatio();
          if(std::abs(diff) > thresh) {
            double adj_factor = (1.0 + diff * dilution);
            for(size_t i = 0; i < dynamic_nodes.size(); i++) {
              dynamic_nodes[i]->setScale(dynamic_nodes[i]->getScale() * adj_factor);
            }
          }
        }
      }
    }

    void run(int iterations, int burn, int thin) {
      for(int i = 1; i <= (iterations + burn); i++) {
        step();
        if(i > burn && (i % thin == 0)) {
          tally();
        }
      }
    }

  public:
    // FIXME: use generic iterators later...
    RMCModel(std::vector<MCMCObject*> mcmcObjects): accepted_(0), rejected_(0), logp_value_(-std::numeric_limits<double>::infinity()), old_logp_value_(-std::numeric_limits<double>::infinity()), mcmcObjects_(mcmcObjects) {
      initChain();
      if(logp()==-std::numeric_limits<double>::infinity()) {
        throw std::logic_error("ERROR: cannot start from a logp of -Inf.");
      }
    }

    double acceptance_ratio() const {
      return accepted_ / (accepted_ + rejected_);
    }

    double logp() const {
      double ans(0);
      //for(auto f : logp_functors) {
      for(std::vector<Likelihiood*>::const_iterator it = logp_functors.begin(); it != logp_functors.end(); it++) {
        ans += (*it)->calc();
      }
      return ans;
    }

    void sample(int iterations, int burn, int adapt, int thin) {
      if(iterations % thin) {
        throw std::logic_error("ERROR: interations not a multiple of thin.");
      }

      // FIXME: kill the magic numbers
      if(adapt >= 200) {
        tune(adapt,static_cast<int>(adapt/100));
      }
      if(true) { tune_global(adapt,static_cast<int>(adapt/100)); }
      run(iterations, burn, thin);
    }
  };
} // namespace cppbugs
#endif // R_MCMC_MODEL_H
