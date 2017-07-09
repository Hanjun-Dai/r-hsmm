#include "rnn.h"
#include "mse_criterion_layer.h"
#include "param_layer.h"
#include "tanh_layer.h"
#include "input_layer.h"
#include "c_add_layer.h"
#include "mvn_diag_nll_criterion_layer.h"
#include "exp_layer.h"

const Dtype pi = 3.14159265358979;
#define sqr(x) ((x) * (x))

RNN::RNN()
{
    w_i2h = w_h2h = nullptr;
	learner = new MomentumSGDLearner<mode, Dtype>(&model, cfg::lr, cfg::momentum, cfg::l2_penalty);
    //learner = new AdamLearner<mode, Dtype>(&model, cfg::lr, cfg::l2_penalty);

	g_input.clear();
	g_label.clear();
	for (int i = 0; i < cfg::bptt; ++i)
	{
		g_label.push_back(new DenseMat<mode, Dtype>());
		g_input.push_back(new DenseMat<mode, Dtype>());
	}

	train_dict["last_hidden"] = &last_hidden_train;
    for (int i = 0; i < cfg::bptt; ++i)
    {        
       	train_dict[fmt::sprintf("signal_input_%d", i)] = g_input[i];
       	train_dict[fmt::sprintf("label_%d", i)] = g_label[i]; 
    }
	
	test_dict["last_hidden"] = &last_hidden_test;
	test_dict["signal_input_0"] = g_input[0];
    test_dict["label_0"] = g_label[0];
}

RNN::RNN(IDiffParam<mode, Dtype>* _w_i2h, IDiffParam<mode, Dtype>* _w_h2h)
        : RNN()
{
    w_i2h = _w_i2h;
    w_h2h = _w_h2h;
}

void RNN::CreateParams()
{
    if (w_i2h)
        this->model.AddParam(this->w_i2h);
    else
        add_diff< LinearParam >(this->model, "w_i2h", cfg::x_dim, cfg::n_hidden, 0, cfg::w_scale);
    if (w_h2h)
        this->model.AddParam(this->w_h2h);
    else
        add_diff< LinearParam >(this->model, "w_h2h", cfg::n_hidden, cfg::n_hidden, 0, cfg::w_scale);
    add_diff< LinearParam >(this->model, "w_mu", cfg::n_hidden, cfg::x_dim, 0, cfg::w_scale);    
    add_diff< LinearParam >(this->model, "w_sigma", cfg::n_hidden, cfg::x_dim, 0, cfg::w_scale);  
}

ILayer<mode, Dtype>* RNN::AddNetBlocks(int time_step, 
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

void RNN::InitNet(NNGraph<mode, Dtype>& gnn, 
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

void RNN::Fit(std::vector< std::pair<Dtype*, int> >& segments)
{		
	// remove too short segments
	for (auto it = segments.begin(); it != segments.end(); )
	{
		if (it->second < cfg::bptt)
			it = segments.erase(it);
		else 
			++it;
	}
	if (segments.size() == 0)
		return;
	SetupDataLoader(segments);

	last_hidden_train.Zeros(batch_size, cfg::n_hidden);

	for (int iter = 0; iter < cfg::max_rnn_iter; ++iter)
	{        
		NextBpttBatch();
		net_train.FeedForward(train_dict, TRAIN);
		auto loss_map = net_train.GetLoss();
	    
        Dtype rmse = 0.0, nll = 0.0;
        for (int t = 0; t < cfg::bptt; ++t)
        {
            nll += loss_map[fmt::sprintf("nll_%d", t)]; 
            rmse += loss_map[fmt::sprintf("mse_%d", t)]; 
        }
        rmse = sqrt(rmse / cfg::bptt / batch_size);
        nll /= cfg::bptt * batch_size;

        if (iter % cfg::report_interval == 0)
        {            
            DenseMat<mode, Dtype> buf;            
            net_train.GetState(fmt::sprintf("mu_%d", cfg::bptt - 1), buf);
            Dtype mu = buf.data[0];
            net_train.GetState(fmt::sprintf("sigma_%d", cfg::bptt - 1), buf);
            std::cerr << "mu: " << mu << " sigma: " << buf.data[0] << std::endl;
            std::cerr << fmt::sprintf("iter: %d\trmse: %.4f\tnll: %.4f", iter, rmse, nll) << std::endl;  
        }
        
   		net_train.BackPropagation();
        learner->Update();
	}
}

void RNN::CalcLogll(Dtype* signal, int len, DenseMat<mode, Dtype>* result)
{
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

void RNN::SetupDataLoader(std::vector< std::pair<Dtype*, int> >& segments)
{
    p_segs = &segments;
    batch_size = segments.size() < cfg::batch_size ? segments.size() : cfg::batch_size;

    index_pool.clear();
            	
    for (unsigned i = 0; i < segments.size(); ++i)
    {
	    index_pool.push_back(i);
    }

    cursors.resize(batch_size);
    for (int i = 0; i < batch_size; ++i)
    {
        cursors[i].first = index_pool.front();
        cursors[i].second = 0;
        index_pool.pop_front();
    }
}

void RNN::ReloadSlot(unsigned batch_idx)    
{
	memset(last_hidden_train.data + last_hidden_train.cols * batch_idx, 0, sizeof(Dtype) * last_hidden_train.cols);
   	
   	index_pool.push_back(cursors[batch_idx].first); 
   	cursors[batch_idx].first = index_pool.front();
   	cursors[batch_idx].second = 0;
   	index_pool.pop_front();
}

void RNN::NextBpttBatch()
{
	auto& segments = *p_segs;
    for (int i = 0; i < this->batch_size; ++i)
    {
        // need to load a new sequences                                   
        if (cursors[i].second + cfg::bptt > segments[cursors[i].first].second)
        {                              
            this->ReloadSlot(i); 
        }
    }	
    int cur_pos;
    for (int j = 0; j < cfg::bptt; ++j)
    {                                  
        auto& feat = g_input[j]->DenseDerived();
        auto& label = g_label[j]->DenseDerived();

        feat.Resize(batch_size, cfg::x_dim);
        label.Resize(batch_size, cfg::x_dim);

        for (int i = 0; i < batch_size; ++i)
        {
        	if (cursors[i].second + j - 1 >= 0)
            {
                cur_pos = cursors[i].second + j - 1;
                memcpy(feat.data + i * cfg::x_dim, 
                       segments[cursors[i].first].first + cur_pos * cfg::x_dim, 
                       sizeof(Dtype) * cfg::x_dim);
            } else
            {
                memset(feat.data + i * cfg::x_dim, 0, sizeof(Dtype) * cfg::x_dim);
            }
        	
            cur_pos = cursors[i].second + j;
            memcpy(label.data + i * cfg::x_dim, 
                   segments[cursors[i].first].first + cur_pos * cfg::x_dim, 
                   sizeof(Dtype) * cfg::x_dim);
        }
    } 
    for (int i = 0; i < this->batch_size; ++i)
        cursors[i].second += cfg::sliding;
}
