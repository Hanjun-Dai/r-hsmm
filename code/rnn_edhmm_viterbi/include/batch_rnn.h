#ifndef BATCH_RNN_H
#define BATCH_RNN_H


#include "i_emit_model.h"
#include "linear_param.h"
#include "nngraph.h"
#include "param_layer.h"
#include "input_layer.h"
#include "c_add_layer.h"
#include "fmt/format.h"
#include "relu_layer.h"
#include "model.h"
#include "mse_criterion_layer.h"
#include "learner.h"
#include <deque>

class BatchRNN : public IEmitModel
{
public:
		BatchRNN();

		virtual void Fit(std::vector< std::pair<Dtype*, int> >& segments) override;
		virtual void CalcLogll(Dtype* signal, int len, DenseMat<mode, Dtype>* result) override;

		NNGraph<mode, Dtype> net_train, net_test;
    	Model<mode, Dtype> model;
    	ExplicitBatchLearner<mode, Dtype>* learner;
		std::vector< IMatrix<mode, Dtype>* > g_input, g_label;	
		DenseMat<mode, Dtype> last_hidden_train, last_hidden_test;

		void SetupNet()
		{
			CreateParams();
			InitNet(net_train, model.all_params, cfg::bptt);
			InitNet(net_test, model.all_params, 1);
		}
		
protected:

		virtual void CreateParams();		

		virtual ILayer<mode, Dtype>* AddNetBlocks(int time_step, 
												  NNGraph<mode, Dtype>& gnn, 
												  ILayer<mode, Dtype> *last_hidden_layer, 
                        		           		  std::map< std::string, IParam<mode, Dtype>* >& param_dict); 
		void InitNet(NNGraph<mode, Dtype>& gnn, 
				std::map< std::string, IParam<mode, Dtype>* >& param_dict, 
                unsigned n_unfold);
		DenseMat<mode, Dtype> buf; 

		std::pair<int, int> cursors;
		std::deque< unsigned > index_pool;
		std::vector< std::pair<Dtype*, int> >* p_segs;		
		void SetupDataLoader(std::vector< std::pair<Dtype*, int> >& segments);
    	void ReloadSlot();

    	void NextBpttBatch(); 

		std::map<std::string, IMatrix<mode, Dtype>* > train_dict, test_dict;					
		int cur_bptt;
};


#endif