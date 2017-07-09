#ifndef I_EMIT_MODEL_H
#define I_EMIT_MODEL_H

#include "config.h"
#include "dense_matrix.h"

class IEmitModel
{
public:
		virtual void Fit(std::vector< std::pair<Dtype*, int> >& segments) = 0;
		virtual void CalcLogll(Dtype* signal, int len, DenseMat<mode, Dtype>* result) = 0;
};

#endif