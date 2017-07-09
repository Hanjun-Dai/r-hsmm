#include "rayleigh.h"
#define min(x, y) (x < y ? x : y)

Rayleigh::Rayleigh(int _order, Dtype _lambda0, Dtype _alpha) 
		: order(_order), lambda0(_lambda0), alpha(_alpha)
{

}

Dtype Rayleigh::logpdf(Dtype x, Dtype sigma)
{
	return log( x / sigma / sigma * exp(-x * x / 2 / sigma / sigma) + 1e-20);
}

void Rayleigh::Fit(std::vector< std::pair<Dtype*, int> >& segments)
{

}

void Rayleigh::CalcLogll(Dtype* signal, int len, DenseMat<mode, Dtype>* result)
{
		result->Resize(1, len);

    	result->data[0] = logpdf(signal[0], lambda0);
    	Dtype lambda, sum = 0;

    	for (int t = 1; t < min(order, len); ++t)
    	{
    		sum += signal[t - 1];
    		lambda = lambda0 + alpha * sum;
    		result->data[t] = logpdf(signal[t], lambda);
    	}

    	for (int t = order; t < len; ++t)
    	{
    		sum += signal[t - 1];
    		lambda = lambda0 + alpha * sum;
    		result->data[t] = logpdf(signal[t], lambda);
    		sum -= signal[t - order];
    	}
}