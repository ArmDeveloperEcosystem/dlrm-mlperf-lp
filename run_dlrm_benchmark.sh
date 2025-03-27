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
dlrm_container="benchmark_dlrm"

# Create results directory
mkdir -p $results_dir/$data_type

###### Run the dlrm container and setup MLPerf #######

# Check if the container exists
echo -e "${yellow}Checking if the container '$dlrm_container' exists...${reset}"
container_exists=$(docker ps -aqf "name=^$dlrm_container$")

if [ -n "$container_exists" ]; then
    echo "${yellow}Container '$dlrm_container' already exists. Will not create a new one. ${reset}"
else
    echo "Creating a new '$dlrm_container' container..."
    docker run -td --shm-size=200G --privileged \
        -v $data_dir:$data_dir \
        -v $model_dir:$model_dir \
        -v $results_dir:$results_dir \
        -e DATA_DIR=$data_dir \
        -e MODEL_DIR=$model_dir \
        -e CONDA_PREFIX=/opt/conda \
        -e NUM_SOCKETS="1" \
        -e CPUS_PER_SOCKET=$(nproc) \
        -e CPUS_PER_PROCESS=$(nproc) \
        -e CPUS_PER_INSTANCE="1" \
        -e CPUS_FOR_LOADGEN="1"  \
        -e BATCH_SIZE="400"  \
        -e PATH=/opt/conda/bin:$PATH \
        --name=$dlrm_container \
        toolsolutions-pytorch:latest
fi



###### Build MLPerf & Dependencies #######

# Copy the patches,  environment.yml file & build script to the benchmark_dlrm container
docker cp $HOME/dlrm-mlperf-lp/. $dlrm_container:$HOME/

echo -e "${yellow}Setting up MLPerf benchmarking inside the container...${reset}"
docker exec -it $dlrm_container bash -c ". $HOME/build_mlperf.sh $data_type"


###### Dump the model #######

dumped_fp32_model="dlrm-multihot-pytorch.pt"
int8_model="aarch64_dlrm_int8.pt"
dlrm_test_path="$HOME/inference_results_v4.0/closed/Intel/code/dlrm-v2-99.9/pytorch-cpu-int8"

# Check if FP32 model is already dumped
if [ -f "$HOME/model/$dumped_fp32_model" ]; then
    echo -e "${yellow}File '$dumped_fp32_model' exists. Skipping model dumping step.${reset}"
else
    echo -e "${yellow}File '$dumped_fp32_model' does not exist. Dumping the model weights...${reset}"
    docker exec -it "$dlrm_container" bash -c " cd $dlrm_test_path && python python/dump_torch_model.py --model-path=$model_dir/model_weights --dataset-path=$data_dir"
fi


###### Calibrate the model #######

# In the case of INT8, calibrate the model if not already calibrated.
echo -e "${yellow}Checking if INT8 model calibration is required...${reset}"

if [ "$data_type" == "int8" ] && [ ! -f "$HOME/model/$int8_model" ]; then
    echo -e "${yellow}File '$int8_model' does not exist. Running calibration...${reset}"
    # the calibration will create aarch64_dlrm_int8.pt in the $HOME/model directory.
    docker exec -it "$dlrm_container" bash -c "cd $dlrm_test_path && ./run_calibration.sh"
else
    echo -e "${yellow}Calibration step is not needed.${reset}"
fi


###### Run the test #######

# Run the offline test
echo -e "${yellow}Running offline test...${reset}"
docker exec -it "$dlrm_container" bash -c "cd $dlrm_test_path && bash run_main.sh offline $data_type"

# Copy results to the host machine
echo -e "${yellow}Copying results to host...${reset}"
docker exec -it "$dlrm_container" bash -c "cd $dlrm_test_path && cp -r output/pytorch-cpu/dlrm/Offline/performance/run_1/* $results_dir/$data_type/"

# Display the MLPerf summary results
echo -e "${yellow}Displaying MLPerf results...${reset}"
cat $results_dir/$data_type/mlperf_log_summary.txt