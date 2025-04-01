#!/bin/bash
 
set -ex
yellow="\e[33m"
reset="\e[0m"
 
data_type=${1:-"int8"}
 
echo -e "${yellow}Data type chosen for the setup is $data_type${reset}"
 
# setup environment variables for the dlrm container
data_dir=$HOME/data/
model_dir=$HOME/model/
results_dir=$HOME/results/
dlrm_setup_dir=$HOME/dlrm-mlperf-lp

export CONDA_PREFIX=/opt/conda
export NUM_SOCKETS="1"
export CPUS_PER_SOCKET=$(nproc)
export CPUS_PER_PROCESS=$(nproc)
export CPUS_PER_INSTANCE="1"
export CPUS_FOR_LOADGEN="1"
export BATCH_SIZE="400"
export PATH=/opt/conda/bin:$PATH 
 
# Create results directory
mkdir -p $results_dir/$data_type

###### Build MLPerf & Dependencies #######
bash -c ". $dlrm_setup_dir/build_mlperf.sh $data_type"
 
 
###### Dump the model #######
 
dumped_fp32_model="dlrm-multihot-pytorch.pt"
int8_model="aarch64_dlrm_int8.pt"
dlrm_test_path="$HOME/inference_results_v4.0/closed/Intel/code/dlrm-v2-99.9/pytorch-cpu-int8"
 
# Check if FP32 model is already dumped
if [ -f "$HOME/model/$dumped_fp32_model" ]; then
    echo -e "${yellow}File '$dumped_fp32_model' exists. Skipping model dumping step.${reset}"
else
    echo -e "${yellow}File '$dumped_fp32_model' does not exist. Dumping the model weights...${reset}"
    bash -c " cd $dlrm_test_path && python python/dump_torch_model.py --model-path=$model_dir/model_weights --dataset-path=$data_dir"
fi
 
 
###### Calibrate the model #######
 
# In the case of INT8, calibrate the model if not already calibrated.
echo -e "${yellow}Checking if INT8 model calibration is required...${reset}"
 
if [ "$data_type" == "int8" ] && [ ! -f "$HOME/model/$int8_model" ]; then
    echo -e "${yellow}File '$int8_model' does not exist. Running calibration...${reset}"
    # the calibration will create aarch64_dlrm_int8.pt in the $HOME/model directory.
    bash -c "cd $dlrm_test_path && ./run_calibration.sh"
else
    echo -e "${yellow}Calibration step is not needed.${reset}"
fi
 
 
###### Run the test #######
 
# Run the offline test
echo -e "${yellow}Running offline test...${reset}"
bash -c "cd $dlrm_test_path && MODEL_DIR=$model_dir DATA_DIR=$data_dir bash run_main.sh offline $data_type"
 
# Copy results to the host machine
echo -e "${yellow}Copying results to host...${reset}"
bash -c "cd $dlrm_test_path && cp -r output/pytorch-cpu/dlrm/Offline/performance/run_1/* $results_dir/$data_type/"
 
# Display the MLPerf summary results
echo -e "${yellow}Displaying MLPerf results...${reset}"
cat $results_dir/$data_type/mlperf_log_summary.txt
