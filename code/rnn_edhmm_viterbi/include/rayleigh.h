#ifndef RAYLEIGH_H
#define RAYLEIGH_H

#include "i_emit_model.h"

class Rayleigh : public IEmitModel
{
public:

		Rayleigh(int _order, Dtype _lambda0, Dtype _alpha);
		virtual void Fit(std::vector< std::pair<Dtype*, int> >& segments) override;
		virtual void CalcLogll(Dtype* signal, int len, DenseMat<mode, Dtype>* result) override;

		Dtype logpdf(Dtype x, Dtype sigma);		

		int order;
		Dtype lambda0, alpha;
};

#endif