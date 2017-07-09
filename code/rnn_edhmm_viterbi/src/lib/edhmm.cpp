#include "edhmm.h"
#include "sequence.h"

MatrixXd EDHMM::p_trans;
MatrixXd EDHMM::p_init;
MatrixXd EDHMM::p_dur;

void EDHMM::Init()
{
		p_trans = MatrixXd::Random(cfg::num_states, cfg::num_states);
		p_init = MatrixXd::Random(1, cfg::num_states);
		p_dur = MatrixXd::Random(cfg::num_states, cfg::max_dur);	

		p_trans = p_trans.array().exp();
		p_init = p_init.array().exp();
		p_dur = p_dur.array().exp();

		// p_init = MatrixXd::Ones(1, cfg::num_states) / cfg::num_states;
		// p_trans = MatrixXd::Zero(cfg::num_states, cfg::num_states);
		// for (int i = 0; i < cfg::num_states; ++i)
		// 	p_trans(i, (i + 1) % cfg::num_states) = 1;
			
		Normalize();
}

void EDHMM::Normalize()
{
		p_init /= p_init.sum();

		for (int i = 0; i < cfg::num_states; ++i)
		{
			p_trans(i, i) = 0;

			p_trans.row(i) /= p_trans.row(i).sum();
			p_dur.row(i) /= p_dur.row(i).sum();
		}
}

void EDHMM::Mstep(std::vector<Sequence*>& dataset)
{
		p_init *= 0;
		p_trans *= 0;
		p_dur *= 0;
		for (size_t i = 0; i < dataset.size(); ++i)
		{
			auto& segments = dataset[i]->code;
			for (size_t j = 0; j < segments.size(); ++j)
			{				
				int cur_state = segments[j].first;
				int cur_dur = segments[j].second - 1;
				p_dur(cur_state, cur_dur) += 1;
				if (j == 0)
					p_init(0, cur_state) += 1;
				if (j + 1 < segments.size())
				{
					int next_state = segments[j + 1].first;
					p_trans(cur_state, next_state) += 1;
				}
			}
		}

		p_init = p_init.array() + 1;
		p_trans = p_trans.array() + 1;
		p_dur = p_dur.array() + 1;

		// p_init = MatrixXd::Ones(1, cfg::num_states) / cfg::num_states;
		// p_trans = MatrixXd::Zero(cfg::num_states, cfg::num_states);
		// for (int i = 0; i < cfg::num_states; ++i)
		// 	p_trans(i, (i + 1) % cfg::num_states) = 1;		

		Normalize();
}
