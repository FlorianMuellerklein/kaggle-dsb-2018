#!/bin/bash

tmux new-session -d -s dsb_training
tmux rename-window 'Training'
tmux send-keys -t dsb_training 'vim *' C-m
tmux split-window -h
tmux resize-pane -R 15
tmux select-pane -t dsb_training:2
tmux split-window -v -t 1
tmux resize-pane -D 15
tmux split-window -v -t 1
tmux -2 attach-session -t dsb_training
