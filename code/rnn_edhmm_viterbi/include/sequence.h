#ifndef SEQUENCE_H
#define SEQUENCE_H

#include "config.h"
#include <vector>
#include <deque>
#include <string>
#include "dense_matrix.h"

#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;

class IEmitModel;
class Sequence
{
public:

		Sequence();
		~Sequence();
		Sequence(const std::vector<Dtype>& _signal, const std::vector<int>& _label);

		Dtype Viterbi(std::vector<IEmitModel*>& emit_models);

		int T;
		std::vector<Dtype> signal;
		std::vector<int> label;
		std::deque< std::pair<int, int> > code;
		
		static Tensor<Dtype, 3> logemit, alpha; 		
		static Tensor<int, 4> pre;
		std::vector<DenseMat<mode, Dtype>* > buffers; 
		std::vector<int> tmp_best_dur;
};

#endif