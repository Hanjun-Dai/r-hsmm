#ifndef cfg_H
#define cfg_H

typedef double Dtype;
#include "fmt/format.h"
#include <set>
#include "imatrix.h"
#include <map>
const MatMode mode = CPU;

struct cfg
{    
    static int dev_id, iter, n_threads;
    static int bptt, sliding, x_dim;
    static unsigned batch_size, n_hidden; 
    static unsigned max_epoch; 
    static int num_states;
    static int max_dur, n_mix;
    static int use_batch_rnn;
    static int max_rnn_iter;
    static int report_interval, test_interval;
    static Dtype w_scale;
    static Dtype lr;
    static Dtype l2_penalty; 
    static Dtype momentum; 
    static const char *save_dir, *label_file, *signal_file;
    
    inline static void LoadParams(const int argc, const char** argv)
    {
        sliding = -1;
        const char* meta_file = nullptr;
        for (int i = 1; i < argc; i += 2)
        {
            if (strcmp(argv[i], "-signal") == 0)
                signal_file = argv[i + 1];
            if (strcmp(argv[i], "-label") == 0)
                label_file = argv[i + 1];
            
            if (strcmp(argv[i], "-s") == 0)
                num_states = atoi(argv[i + 1]);
	        if (strcmp(argv[i], "-thread") == 0)
		        n_threads = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-brnn") == 0)
                use_batch_rnn = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-sliding") == 0)
                sliding = atoi(argv[i + 1]);            
            if (strcmp(argv[i], "-rnn_iter") == 0)
                max_rnn_iter = atoi(argv[i + 1]);            
            if (strcmp(argv[i], "-d") == 0)
                max_dur = atoi(argv[i + 1]);
		    if (strcmp(argv[i], "-lr") == 0)
		        lr = atof(argv[i + 1]);
            if (strcmp(argv[i], "-int_report") == 0)
                report_interval = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-int_test") == 0)
                test_interval = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-bptt") == 0)
                bptt = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-cur_iter") == 0)
                iter = atoi(argv[i + 1]);
		    if (strcmp(argv[i], "-hidden") == 0)
			    n_hidden = atoi(argv[i + 1]);             
		    if (strcmp(argv[i], "-b") == 0)
    			batch_size = atoi(argv[i + 1]);
    		if (strcmp(argv[i], "-maxe") == 0)
	       		max_epoch = atoi(argv[i + 1]);
    		if (strcmp(argv[i], "-l2") == 0)
    			l2_penalty = atof(argv[i + 1]);
    		if (strcmp(argv[i], "-m") == 0)
    			momentum = atof(argv[i + 1]);	
    		if (strcmp(argv[i], "-svdir") == 0)
    			save_dir = argv[i + 1];
            if (strcmp(argv[i], "-meta") == 0)
                meta_file = argv[i + 1];
            if (strcmp(argv[i], "-device") == 0)
    			dev_id = atoi(argv[i + 1]);
        }	
        assert(meta_file);
        FILE* fid = fopen(meta_file, "r");
        fscanf(fid, "%d %d", &x_dim, &num_states);
        fclose(fid);
        if (sliding < 0 || sliding > bptt)
            sliding = bptt;
        std::cerr << "use_batch_rnn = " << use_batch_rnn << std::endl;
	    std::cerr << "n_threads = " << n_threads << std::endl;
        std::cerr << "sliding = " << sliding << std::endl;
        std::cerr << "max_rnn_iter = " << max_rnn_iter << std::endl;
        std::cerr << "num_states = " << num_states << std::endl;
        std::cerr << "max_dur = " << max_dur << std::endl;
        std::cerr << "bptt = " << bptt << std::endl;
	    std::cerr << "n_hidden = " << n_hidden << std::endl;
        std::cerr << "batch_size = " << batch_size << std::endl;
        std::cerr << "max_epoch = " << max_epoch << std::endl;
    	std::cerr << "lr = " << lr << std::endl;
        std::cerr << "l2_penalty = " << l2_penalty << std::endl;
    	std::cerr << "momentum = " << momentum << std::endl;
    	std::cerr << "init iter = " << iter << std::endl;	
        std::cerr << "device id = " << dev_id << std::endl; 
        std::cerr << "report_interval = " << report_interval << std::endl;   
        std::cerr << "test_interval = " << test_interval << std::endl;   
    }    
};

#endif
