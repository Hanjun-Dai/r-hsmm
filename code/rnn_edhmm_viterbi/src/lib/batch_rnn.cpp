#include "batch_rnn.h"
#include "mse_criterion_layer.h"
#include "param_layer.h"
#include "input_layer.h"
#include "c_add_layer.h"
#include "mvn_diag_nll_criterion_layer.h"
#include "exp_layer.h"

const Dtype pi = 3.14159265358979;
#define sqr(x) ((x) * (x))

BatchRNN::BatchRNN()
{
	learner = new ExplicitBatchLearner<mode, Dtype>(&model, cfg::lr, cfg::l2_penalty);

	g_input.clear();
	g_label.clear();
	for (int i = 0; i < cfg::bptt; ++i)
	{
		g_label.push_back(new DenseMat<mode, Dtype>());
		g_input.push_back(new DenseMat<mode, Dtype>());
	}	
}

void BatchRNN::CreateParams()
{
    add_diff< LinearParam >(this->model, "w_i2h", cfg::x_dim, cfg::n_hidden, 0, cfg::w_scale);
    add_diff< LinearParam >(this->model, "w_h2h", cfg::n_hidden, cfg::n_hidden, 0, cfg::w_scale);
    add_diff< LinearParam >(this->model, "w_mu", cfg::n_hidden, cfg::x_dim, 0, cfg::w_scale);    
    add_diff< LinearParam >(this->model, "w_sigma", cfg::n_hidden, cfg::x_dim, 0, cfg::w_scale);  
}

ILayer<mode, Dtype>* BatchRNN::AddNetBlocks(int time_step, 
						  			   NNGraph<mode, Dtype>& gnn, 
									   ILayer<mode, Dtype> *last_hidden_layer, 
                                       std::map< std::string, IParam<mode, Dtype>* >& param_dict)
{		        
	auto* signal_input_layer = cl< InputLayer >(fmt::sprintf("signal_input_%d", time_step), gnn, {});
    auto* signal_label_layer = cl< InputLayer >(fmt::sprintf("label_%d", time_step), gnn, {});

	auto* hidden_layer = cl< ParamLayer >(gnn, {signal_input_layer, last_hidden_layer}, {param_dict["w_i2h"], param_dict["w_h2h"]}); 

    auto* relu_hidden_layer = cl< ReLULayer >(fmt::sprintf("recurrent_hidden_%d", time_step), gnn, {hidden_layer});

    auto* mu_layer = cl< ParamLayer >(fmt::sprintf("mu_%d", time_step), gnn, {relu_hidden_layer}, {param_dict["w_mu"]});
    
    auto* sigma_linear_layer = cl< ParamLayer >(gnn, {relu_hidden_layer}, {param_dict["w_sigma"]});

    auto* sigma_layer = cl< ExpLayer >(fmt::sprintf("sigma_%d", time_step), gnn, {sigma_linear_layer});

    cl< MVNDianNLLCriterionLayer >(fmt::sprintf("nll_%d", time_step), gnn, {mu_layer, sigma_layer, signal_label_layer});    
    cl< MSECriterionLayer >(fmt::sprintf("mse_%d", time_step), gnn, {mu_layer, signal_label_layer}, PropErr::N); 

    return relu_hidden_layer; 
}

void BatchRNN::InitNet(NNGraph<mode, Dtype>& gnn, 
				  std::map< std::string, IParam<mode, Dtype>* >& param_dict, 
                  unsigned n_unfold)
{
    auto* last_hidden_layer = cl< InputLayer >("last_hidden", gnn, {});

    for (unsigned i = 0; i < n_unfold; ++i)
    {
        auto* new_hidden = AddNetBlocks(i, gnn, last_hidden_layer, param_dict);
        last_hidden_layer = new_hidden;
    }
} 

void BatchRNN::Fit(std::vector< std::pair<Dtype*, int> >& segments)
{		
	if (segments.size() == 0)
		return;
    std::map<std::string, IMatrix<mode, Dtype>* > train_dict;
	SetupDataLoader(segments);

	last_hidden_train.Zeros(1, cfg::n_hidden);

	for (int iter = 0; iter < cfg::max_rnn_iter; ++iter)
	{       
        Dtype rmse = 0.0, nll = 0.0, n_samples = 0;

        for (size_t b = 0; b < cfg::batch_size; ++b)
        {
            NextBpttBatch();
            train_dict.clear();
            train_dict["last_hidden"] = &last_hidden_train;
            for (int i = 0; i < cur_bptt; ++i)
            {        
                train_dict[fmt::sprintf("signal_input_%d", i)] = g_input[i];
                train_dict[fmt::sprintf("label_%d", i)] = g_label[i]; 
            }
            net_train.FeedForward(train_dict, TRAIN);
            auto loss_map = net_train.GetLoss();
        
            n_samples += cur_bptt;    
            for (int t = 0; t < cur_bptt; ++t)
            {
                nll += loss_map[fmt::sprintf("nll_%d", t)]; 
                rmse += loss_map[fmt::sprintf("mse_%d", t)]; 
            }
            net_train.BackPropagation();

            learner->AccumulateGrad();            
        }
		rmse = sqrt(rmse / n_samples);
        nll /= n_samples;
        if (iter % cfg::report_interval == 0)
        {            
            std::cerr << fmt::sprintf("iter: %d\trmse: %.4f\tnll: %.4f", iter, rmse, nll) << std::endl;  
        }        
   		
        learner->Update();
	}
}

void BatchRNN::CalcLogll(Dtype* signal, int len, DenseMat<mode, Dtype>* result)
{
    std::map<std::string, IMatrix<mode, Dtype>* > test_dict;    
    test_dict.clear();
    test_dict["last_hidden"] = &last_hidden_test;
    test_dict["signal_input_0"] = g_input[0];
    test_dict["label_0"] = g_label[0];
	result->Zeros(1, len);

	last_hidden_test.Zeros(1, cfg::n_hidden);
	auto& input = g_input[0]->DenseDerived();
    auto& label = g_label[0]->DenseDerived();
    input.Resize(1, cfg::x_dim);
    label.Resize(1, cfg::x_dim);

	for (int i = 0; i < len; ++i)
	{
		if (i == 0)
			input.Zeros();
		else
        {
            memcpy(input.data, signal + (i - 1) * cfg::x_dim, sizeof(Dtype) * cfg::x_dim);
        }
        memcpy(label.data, signal + i * cfg::x_dim, sizeof(Dtype) * cfg::x_dim);

		net_test.FeedForward(test_dict, TEST);
        auto loss_map = net_test.GetLoss();
        result->data[i] = -loss_map["nll_0"];

		net_test.GetState("recurrent_hidden_0", last_hidden_test);		        
	}
}

//== data loader ==

void BatchRNN::SetupDataLoader(std::vector< std::pair<Dtype*, int> >& segments)
{
    p_segs = &segments;
    index_pool.clear();
            	
    for (unsigned i = 0; i < segments.size(); ++i)
    {
	    index_pool.push_back(i);
    }

    cursors.first = index_pool.front();
    cursors.second = 0;
    index_pool.pop_front();
}

void BatchRNN::ReloadSlot()    
{
	memset(last_hidden_train.data, 0, sizeof(Dtype) * last_hidden_train.cols);
   	
   	index_pool.push_back(cursors.first); 
   	cursors.first = index_pool.front();
   	cursors.second = 0;
   	index_pool.pop_front();
}

void BatchRNN::NextBpttBatch()
{
	auto& segments = *p_segs;
    if (cursors.second >= segments[cursors.first].second)
        ReloadSlot();

    cur_bptt = cfg::bptt;    
    if (cursors.second + cfg::bptt > segments[cursors.first].second)
        cur_bptt = segments[cursors.first].second - cursors.second;

    int cur_pos;
    for (int j = 0; j < cur_bptt; ++j)
    {
        auto& feat = g_input[j]->DenseDerived();
        auto& label = g_label[j]->DenseDerived();

        feat.Resize(1, cfg::x_dim);
        label.Resize(1, cfg::x_dim);

        if (cursors.second + j - 1 >= 0)
        {
            cur_pos = cursors.second + j - 1;
            memcpy(feat.data, 
                   segments[cursors.first].first + cur_pos * cfg::x_dim, 
                   sizeof(Dtype) * cfg::x_dim);
        } else
        {
            memset(feat.data, 0, sizeof(Dtype) * cfg::x_dim);
        }        

        cur_pos = cursors.second + j;
        memcpy(label.data, 
               segments[cursors.first].first + cur_pos * cfg::x_dim, 
               sizeof(Dtype) * cfg::x_dim);
    }

    cursors.second += cfg::sliding;    
}
