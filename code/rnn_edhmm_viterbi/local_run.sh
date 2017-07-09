#!/bin/bash

dataset=gp
data_root=../../data/$dataset
RESULT_ROOT=$HOME/scratch/results/rnn_hsmm/$dataset

num_states=3
max_dur=100
n_hidden=128
max_rnn_iter=20000
bsize=50
learning_rate=0.0001
max_iter=2000000
cur_iter=0
dev_id=0
bptt=3
int_report=5000
int_test=1
brnn=0
sliding=$bptt
save_dir=$RESULT_ROOT/riter-$max_rnn_iter-hidden-$n_hidden-bsize-$bsize-bptt-$bptt

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi


build/main \
        -brnn $brnn \
        -thread 3 \
        -sliding $sliding \
        -rnn_iter $max_rnn_iter \
        -bptt $bptt \
        -meta $data_root/meta.txt \
        -s $num_states \
        -d $max_dur \
        -label $data_root/labels.txt \
        -signal $data_root/signals.txt \
        -lr $learning_rate \
        -device $dev_id \
        -maxe $max_iter \
        -svdir $save_dir \
        -hidden $n_hidden \
        -int_report $int_report \
        -int_test $int_test \
        -l2 0.00 \
        -m 0.0 \
        -b $bsize \
        -cur_iter $cur_iter \
        2>&1 | tee $save_dir/log.txt 
