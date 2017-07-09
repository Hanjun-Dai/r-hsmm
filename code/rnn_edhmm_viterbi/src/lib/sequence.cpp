#include "sequence.h"
#include <cstring>
#include "rnn.h"
#include "edhmm.h"
#include "timer.h"

#define min(x, y) (x < y ? x : y)

Tensor<Dtype, 3> Sequence::logemit;
Tensor<Dtype, 3> Sequence::alpha; 		
Tensor<int, 4> Sequence::pre;

Sequence::Sequence()
{

}

Sequence::~Sequence()
{
}

Sequence::Sequence(const std::vector<Dtype>& _signal, const std::vector<int>& _label)
		: signal(_signal), label(_label)
{
	code.clear();
	assert(signal.size() == cfg::x_dim * label.size());
	T = label.size();

	buffers.clear();
	for (int i = 0; i < cfg::num_states; ++i)
	{
		buffers.push_back(new DenseMat<mode, Dtype>());
	}
	tmp_best_dur.resize(cfg::num_states);
}

Dtype Sequence::Viterbi(std::vector<IEmitModel*>& emit_models)
{
	alpha.resize(T, cfg::num_states, cfg::max_dur);
	logemit.resize(cfg::num_states, T, cfg::max_dur);	
	pre.resize(T, cfg::num_states, cfg::max_dur, 2);	
	alpha.setZero();
	logemit.setZero();
	// init

	Timer::Tic();
	//std::cerr << "pre-processing.... ";
	#pragma omp parallel for
	for (int s = 0; s < cfg::num_states; ++s)
	{		
		for (int t = 0; t < T; ++t)
		{
			int max_dur = min(cfg::max_dur, T - t);
			emit_models[s]->CalcLogll(signal.data() + t * cfg::x_dim, max_dur, buffers[s]);
			if (t == 0)
			{
				Dtype sum = 0;

				for (int d = 0; d < cfg::max_dur; ++d)
				{
					sum += buffers[s]->data[d];
					alpha(0, s, d) = log(EDHMM::p_init(0, s)) + log(EDHMM::p_dur(s, d)) + sum;
				}
			}
			for (int d = 0; d < max_dur; ++d)
			{
				logemit(s, t + d, d) = buffers[s]->data[d];
			}
		}
	}

	Timer::Toc();
	Timer::Tic();
	//std::cerr << "dp... ";
	// dp	
	for (int t = 1; t < T; ++t)
	{		
		for (int i = 0; i < cfg::num_states; ++i)
		{
			tmp_best_dur[i] = 0;
			Dtype tmp_best = alpha(t - 1, i, 0);
			for (int dd = 1; dd < min(cfg::max_dur, t); ++dd)
			{
				if (alpha(t - 1, i, dd) > tmp_best)
				{
					tmp_best = alpha(t - 1, i, dd);
					tmp_best_dur[i] = dd;
				}
			}
		}

		#pragma omp parallel for
		for (int j = 0; j < cfg::num_states; ++j)
		{
			for (int d = 0; d < min(cfg::max_dur, t + 1); ++d)
			{
				if (d)
				{
					alpha(t, j, d) = alpha(t - 1, j, d - 1) - log(EDHMM::p_dur(j, d - 1)) + log(EDHMM::p_dur(j, d)); 
					pre(t, j, d, 0) = pre(t - 1, j, d - 1, 0);
					pre(t, j, d, 1) = pre(t - 1, j, d - 1, 1);
				} else 
				{
					bool got_one = false;
					Dtype tmp_best = 0;
					for (int i = 0; i < cfg::num_states; ++i)
					{
						if (i == j || fabs(EDHMM::p_trans(i, j)) < 1e-6 )
							continue;						
						Dtype score = alpha(t - 1, i, tmp_best_dur[i]) + log(EDHMM::p_trans(i, j));
						if (!got_one || score > tmp_best)
						{
							tmp_best = score;
							got_one = true;
							pre(t, j, d, 0) = i;
							pre(t, j, d, 1) = tmp_best_dur[i];
						}
					}
					alpha(t, j, d) = tmp_best + log(EDHMM::p_dur(j, 0));
				}
				alpha(t, j, d) += logemit(j, t, d);							
			}
		}		
	}
	Timer::Toc();
	// construct solution
	bool got_one = false;
	Dtype cur_best = 0;
	int last_d = 0, last_j = 0;
	for (int j = 0; j < cfg::num_states; ++j)
		for (int d = 0; d < cfg::max_dur; ++d)
			if (!got_one || alpha(T - 1, j, d) > cur_best)
			{
				cur_best = alpha(T - 1, j, d);
				got_one = true;
				last_d = d;
				last_j = j;
			}

	code.clear();
	int t = T - 1;
	while (t >= 0)
	{
		code.push_front(std::make_pair(last_j, last_d + 1));
		int prev_j = pre(t, last_j, last_d, 0);
		int prev_d = pre(t, last_j, last_d, 1);
		t -= last_d + 1;
		last_j = prev_j;
		last_d = prev_d;
	}

	return cur_best;
}

