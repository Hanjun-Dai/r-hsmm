# r-hsmm
Implementation of Recurrent Hidden Semi-Markov Model   http://www.cc.gatech.edu/~lsong/papers/DaiDaiZhaLietal17.pdf

# build

Get the source code

    git clone https://github.com/Hanjun-Dai/r-hsmm
    
This code depends on an obsolete graphnn library: 

    cd code/graphnn-1.11
    build the graphnn library with the instructions here:
    https://github.com/Hanjun-Dai/graphnn/tree/c59a2dd15cf528bfe0ade5a5680466dfaf027c0a
    
Build the rnn-hsmm c++ code:

    cd code/rnn_edhmm_viterbi
    make

# learning

  #### lambda = 0
  essentially this is equivalent to EM. 
  
  #### lambda = \infty
  
  coming soon
  
# inference

  One can use either EM or variational auto-encoder to do inference. 
  
  #### EM: 
  
    use the viterbi decoder in code/rnn_edhmm_viterbi
    
  #### VAE: 
  
    coming soon 
