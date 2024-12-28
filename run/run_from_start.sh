#!/bin/bash





data_seed=888  # Experiment seed
percentage=0.05   # Data percentage to select
model_load_path=meta-llama/Meta-Llama-3.1-8B    # Your base LLM architechture. Here, we take meta-llama/Meta-Llama-3.1-8B as an example
devices="0 1 2 3" # Cuda devices available to use
max_collect_samples=None  # The number of training data you want to test the code, after everything works, you can set it to None to run on all training data
projection_dims=8192  # The projection dimension

./run_rose.sh "$data_seed" "$percentage" "$model_load_path" "$devices" "$max_collect_samples" "$projection_dims"
