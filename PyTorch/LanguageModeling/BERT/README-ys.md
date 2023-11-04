docker build \
  --network=host \
  --build-arg http_proxy=http://172.22.2.31:7000 --build-arg https_proxy=http://172.22.2.31:7000 \
  --build-arg FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.11-py3 \
  --rm \
  --pull \
  --no-cache \
  -t deeplearning_test_pytorch:21.11-ys-3 .



 docker run -it -d --name bert_test \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --net=host \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v $PWD:/workspace/bert \
  -v $PWD/results:/results \
  deeplearning_test_pytorch:21.11-ys-3 bash
 

base cased bert
train_batch_size=${1:-8192}
learning_rate=${2:-"6e-3"}
precision=${3:-"fp16"}







on base bert
SQUAD fp16:
DLL 2023-11-04 03:36:56.678414 -  e2e_train_time : 342.629679441452 s training_sequences_per_second : 517.415771713068 sequences/s final_loss : 0.08813073486089706 None
DLL 2023-11-04 03:36:56.678890 -  e2e_inference_time : 25.27370309829712 s inference_sequences_per_second : 428.627334817821 sequences/s
DLL 2023-11-04 03:36:56.679024 -  exact_match : 80.65279091769158 None F1 : 88.12721039728963 None
SQUAD TF32:
DLL 2023-11-04 06:01:24.299646 -  e2e_train_time : 737.3864576816559 s training_sequences_per_second : 240.41938681295406 sequences/s final_loss : 0.0877014771103859 None
DLL 2023-11-04 06:01:24.300171 -  e2e_inference_time : 43.09560489654541 s inference_sequences_per_second : 251.3713411380469 sequences/s
DLL 2023-11-04 06:01:24.300324 -  exact_match : 80.66225165562913 None F1 : 88.0898698775224 None


# triton

./triton/large/scripts/docker/build.sh ->

docker build   --network=host \
  --build-arg FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.11-py3 \
  --build-arg http_proxy=http://192.168.107.122:7000 --build-arg https_proxy=http://192.168.107.122:7000 \
  -t bert-triton:ys-1 . -f triton/Dockerfile


docker tag bert-triton:ys-1 gpu02:5000/bert-triton:21.11-ys-3

docker run -it -d --name bert_test_triton \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --net=host \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ipc=host \
  -e WORKDIR="$(pwd)" \
  -e PYTHONPATH="$(pwd)" \
  -v "$(pwd)":"$(pwd)" \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -w "$(pwd)" \
  gpu02:5000/bert-triton:21.11-ys-3 bash


docker exec -ti bert_test_triton bash
./triton/large/runner/prepare_datasets.sh


*** Measurement Settings ***
  Batch size: 16
  Using "count_windows" mode for stabilization
  Minimum number of samples in each window: 50
  Using synchronous calls for inference
  Stabilizing using average latency

```shell
python triton/run_performance_on_triton.py \
    --model-repository ${MODEL_REPOSITORY_PATH} \
    --model-name ${MODEL_NAME} \
    --input-data ${SHARED_DIR}/input_data/data.json \
    --input-shapes input__0:${MAX_SEQ_LENGTH} \
    --input-shapes input__1:${MAX_SEQ_LENGTH} \
    --input-shapes input__2:${MAX_SEQ_LENGTH} \
    --batch-sizes ${BATCH_SIZE} \
    --number-of-triton-instances ${TRITON_INSTANCES} \
    --number-of-model-instances ${TRITON_GPU_ENGINE_COUNT} \
    --batching-mode static \
    --evaluation-mode offline \
    --performance-tool perf_analyzer \
    --result-path ${SHARED_DIR}/triton_performance_offline.csv
```

```bash
Request concurrency: 1
  Client:
    Request count: 50
    Throughput: 266.667 infer/sec
    Avg latency: 59227 usec (standard deviation 375 usec)
    p50 latency: 59260 usec
    p90 latency: 59754 usec
    p95 latency: 59885 usec
    p99 latency: 60101 usec
    Avg gRPC time: 59215 usec ((un)marshal request/response 16 usec + response wait 59199 usec)
  Server:
    Inference count: 800
    Execution count: 50
    Successful request count: 50
    Avg request latency: 58912 usec (overhead 53 usec + queue 58 usec + compute input 151 usec + compute infer 58639 usec + compute output 11 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 266.667 infer/sec, latency 59227 usec
  Batch    Concurrency    Inferences/Second    Client Send    Network+Server Send/Recv    Server Queue    Server Compute Input    Server Compute Infer    Server Compute Output    Client Recv    p50 latency    p90 latency    p95 latency    p99 latency    avg latency
-------  -------------  -------------------  -------------  --------------------------  --------------  ----------------------  ----------------------  -----------------------  -------------  -------------  -------------  -------------  -------------  -------------
     16              1              266.667             14                         349              58                     151                   58639                       11              2          59260          59754          59885          60101          59224
```

https://www.cnblogs.com/zzk0/p/15543824.html
https://github.com/triton-inference-server/model_analyzer/blob/main/docs/cli.md