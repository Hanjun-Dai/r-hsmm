#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "config.h"
#include "sequence.h"
#include "rnn.h"
#include "rayleigh.h"
#include "edhmm.h"
#include "batch_rnn.h"
#include "timer.h"
#include "km.h"

#define EIGEN_DONT_PARALLELIZE

std::vector<Sequence*> dataset;
std::vector<IEmitModel*> emit_models;
BipartiteGraph* bg;

void LoadSequences(std::vector<Sequence*>& dataset, const char* signal_file, const char* label_file)
{
	dataset.clear();

	std::ifstream s_signal(signal_file), s_label(label_file);

	int num_seqs;
	s_signal >> num_seqs;
	
	std::vector<Dtype> vec_sig;
	std::vector<int> vec_label;
	for (int i = 0; i < num_seqs; ++i)
	{
		Dtype sig; int ll, len;
		s_signal >> len;
		vec_sig.clear();
		vec_label.clear();
		for (int j = 0; j < len; ++j)
		{
			for (int k = 0; k < cfg::x_dim; ++k)
			{
				s_signal >> sig;
				vec_sig.push_back(sig);
			}
			s_label >> ll;
			vec_label.push_back(ll);
		}

		Sequence* seq = new Sequence(vec_sig, vec_label);
		dataset.push_back(seq);
	}	
}

void GetSegments(std::vector<Sequence*>& dataset, int state, std::vector< std::pair< Dtype*, int > >& segments)
{
	segments.clear();
	for (size_t i = 0; i < dataset.size(); ++i)
	{
		auto* seq = dataset[i];
		int cur_pos = 0;
		for (size_t j = 0; j < seq->code.size(); ++j)
		{			
			if (seq->code[j].first == state)
			{
				segments.push_back(std::make_pair(seq->signal.data() + cur_pos * cfg::x_dim, seq->code[j].second)); 
			}
			cur_pos += seq->code[j].second;
		}
	}
}

void GetTrueSegments(std::vector<Sequence*>& dataset, int state, std::vector< std::pair< Dtype*, int > >& segments)
{
	segments.clear();
	for (size_t i = 0; i < dataset.size(); ++i)
	{
		auto* seq = dataset[i];
		int cur_pos = -1, cur_state = -1;
		for (size_t j = 0; j < seq->label.size(); ++j)
		{
			if (seq->label[j] != cur_state) // new segment
			{
				if (cur_state == state)
				{
					segments.push_back(std::make_pair(seq->signal.data() + cur_pos * cfg::x_dim, j - cur_pos));
				}
				cur_pos = j;
				cur_state = seq->label[j];
			}
		}
		if (cur_state == state)
		{
			segments.push_back(std::make_pair(seq->signal.data() + cur_pos * cfg::x_dim, seq->label.size() - cur_pos));
		}
	}
}

Dtype EvaluateAcc(std::vector<Sequence*>& dataset, int cur_round, std::string phase, int seq_idx)
{
	for (int i = 0; i < cfg::num_states; ++i)
		for (int j = 0; j < cfg::num_states; ++j)
			bg->w[i][j] = 0;

	auto* seq = dataset[seq_idx];
	for (int state = 0; state < cfg::num_states; ++state)
	{			
			int cur_pos = 0;
			for (size_t j = 0; j < seq->code.size(); ++j)
			{	
				for (int k = 0; k < seq->code[j].second; ++k)
					if (seq->label[cur_pos + k] == state)
						bg->w[state][seq->code[j].first]++;

				cur_pos += seq->code[j].second;
			}			
			assert(cur_pos == (int)seq->label.size());	
	}

	bg->Init();
	Dtype acc = bg->BestMatch();
	std::vector<int> best_map;
	best_map.resize(cfg::num_states);

	for (int i = 0; i < cfg::num_states; ++i)
		best_map[i] = bg->nlink[i];

	Dtype total_samples = seq->label.size();
	
	FILE* fid = fopen(fmt::sprintf("%s/%s_seg_iter_%d_seq_%d.txt", cfg::save_dir, phase.c_str(), cur_round, seq_idx).c_str(), "w");
        
                for (size_t j = 0; j < seq->code.size(); ++j)
                {
                        for (int k = 0; k < seq->code[j].second; ++k)
                        {
                                fprintf(fid, "%d\n", best_map[seq->code[j].first]);
                        }
                }        
        fclose(fid);

	return acc / total_samples;        
}

void InitRNN()
{
	for (int i = 0; i < cfg::num_states; ++i)
	{
		auto* model = new RNN();
		model->SetupNet();
		emit_models.push_back(model);
	}
}

void TrainRNN()
{
	InitRNN();

	for (int iter = 0; iter < (int)cfg::max_epoch; ++iter)
	{
		// E-step				
		std::cerr << "E-step" << std::endl;
		Dtype ll = 0.0, acc = 0;
		for (size_t i = 0; i < dataset.size(); ++i)
		{
			if (i % 1 == 0)
				std::cerr << i << std::endl;
			ll +=  dataset[i]->Viterbi(emit_models);
			acc += EvaluateAcc(dataset, iter, "viterbi", i);
		}			
		std::cerr << "accuracy: " << acc / dataset.size();		
		printf("em_iter: %d\tll: %.4f\n", iter, ll);

		// M-step 
		// estimate A, pi, dur
		printf("\n\n\n");
		std::cerr << "==========M-step=======" << std::endl;
		EDHMM::Mstep(dataset);		
		
		// train RNN
		#pragma omp parallel for
		for (int s = 0; s < cfg::num_states; ++s)
		{
			std::vector< std::pair< Dtype*, int > > segments;
			std::cerr << "training state " << s << std::endl;
			GetSegments(dataset, s, segments);
			emit_models[s]->Fit(segments);			
		}
		std::cerr << "==========done M-step=======" << std::endl;
		printf("\n\n\n");
	}
}

int main(const int argc, const char** argv)
{
	srand(time(NULL));
	cfg::LoadParams(argc, argv);
	
	Eigen::initParallel();
	mkl_set_num_threads(1);
	omp_set_num_threads(cfg::n_threads);
	

	GPUHandle::Init(cfg::dev_id);
	EDHMM::Init();
	bg = new BipartiteGraph(cfg::num_states);

	LoadSequences(dataset, cfg::signal_file, cfg::label_file);
	TrainRNN();

	GPUHandle::Destroy();
	return 0;
}
