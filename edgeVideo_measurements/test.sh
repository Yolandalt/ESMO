NUM_THREADS=5
for thread_id in $(seq 0 $(($NUM_THREADS - 1))); do
    echo $thread_id
    # tmux send-keys -t $TMUX_SESSION_NAME:thread-$thread_id "conda deactivate" \
    #                                         " $@" Enter
    # tmux send-keys -t $TMUX_SESSION_NAME:thread-$thread_id "conda activate mec" \
    #                                             " $@" Enter       
    # tmux send-keys -t $TMUX_SESSION_NAME:thread-$thread_id \
    #     "python3 thread.py" \
    #     " --env_name=$ENV_NAME" \
    #     " --log_dir=$LOG_DIR" \
    #     " --num_threads=$NUM_THREADS" \
    #     " --worker_index=$thread_id" \
    #     " $@" Enter
done
