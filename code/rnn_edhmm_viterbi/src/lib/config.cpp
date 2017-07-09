#include "config.h"

int cfg::n_threads = 1;
int cfg::sliding = -1;
int cfg::dev_id = 0;
int cfg::iter = 0;
int cfg::x_dim = 0;
int cfg::max_dur = 0;
int cfg::num_states = 0;
int cfg::max_rnn_iter = 0;
int cfg::bptt = 3;
int cfg::n_mix = 1;
int cfg::report_interval = 100;
int cfg::test_interval = 1;
unsigned cfg::n_hidden = 256;
unsigned cfg::batch_size = 50;
unsigned cfg::max_epoch = 200;
int cfg::use_batch_rnn = 0;

Dtype cfg::lr = 0.0005;
Dtype cfg::l2_penalty = 0;
Dtype cfg::momentum = 0;
Dtype cfg::w_scale = 0.01;
const char* cfg::save_dir = "./saved";
const char* cfg::signal_file = nullptr;
const char* cfg::label_file = nullptr;