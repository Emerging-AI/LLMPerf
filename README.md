# LLMPerf: In-Depth Performance Analysis of LLM Services on GPU Cloud Environments

LLMPerf is an open-source project designed to deliver comprehensive, ready-to-use insights into the performance of Large Language Model (LLM) services. 
Through in-depth performance evaluations, LLMPerf provides benchmarking results that developers, researchers, and engineers can immediately use to understand the effects of different deployment strategies on LLM performance.

This project focuses on key performance metrics such as latency, throughput, and resource utilization. 
LLMPerf’s data-driven insights are particularly valuable for optimizing large-scale LLM serving environments, ensuring efficiency and stability in production settings.


## Introduction

LLMPerf evaluates the performance of LLM services across various hardware configurations and inference scenarios, offering a clear, real-world analysis of different workloads.

#### Experimental Setup

* **Model**
  * Qwen2-7B
  * Qwen2-72B-AWQ
* **GPU**
  * 4090 24 GB × 8, PCIe Gen4.0
  * A100 80 GB × 8, NVLink, PCIe Gen4.0
* **Testing Task**
  * MC (Multiple Choice) TEST: Focuses on small prompt and generation tokens, with minimal network overhead.
  * TKG (Temporal Knowledge Graph Extraction): Handles large prompt and generation tokens, stressing gpu memory for kv cache and in-host network capacity.
* **Request Injection**: Poisson Distribution
* **Injection Time**: 15 min per experiment

#### Evaluation Metric

* The **execution time** distribution: This metric monitors the distribution of request execution times. Anomalies in this distribution indicate that the LLM service may be struggling to process requests efficiently, revealing performance bottlenecks under high loads.

This setup allows us to examine the behavior of LLM services across different usage patterns and GPU infrastructures, with a focus on GPU utilization, inference framework performance, and network bottlenecks.


## In-Depth Performance Analysis of LLM Services

### 1. GPU Performance Analysis

The GPU analysis Focuses on the following factors:

* **Computing Capacity**: The computing capacity difference of GPUs root in FP64/FP32/BF16/FP16/INT8 tensor cores.
* **GPU Memory and Bandwidth**: 
  * 4090: 24 GB Memory and 1 TB/s Bandwidth.
  * A100: 80 GB Memory and 2 TB/s Bandwidth.
* **Interconnect**:
  * 4090: PCIe Gen4 2 GB/s × 16 (32 GB/s per GPU)
  * A100: NVLink 25 GB/s × 12 (300 GB/s per GPU)

#### 1.1 Performance Inside GPU

To evaluate the performance within a single GPU, **Qwen2-7B** was deployed on one GPU (with tensor parallel size = 1), and service performance was tested using requests from the MC Test and TKG datasets.

For MC Test dataset, Qwen2-7B deployed on an A100 processed **1.6× more requests** than on a 4090. 
This difference is due to the superior computing capacity and higher GPU bandwidth of the A100.

For TKG dataset, Qwen2-7B deployed on an A100 processed **2× to 3× more requests** than on a 4090. 
The 4090’s limited GPU memory results in an inability to store cached Key and Value tokens, reducing the number of simultaneous requests.
This leads to under-utilization of the 4090’s computing capacity.

The performance difference is visually presented below:

<img src="./assets/GPU/performance_inside_gpu.png" width="500">


#### 1.2 Performance of Interconnected GPUs




### 2. Inference Framework Performance Analysis

#### 2.1 Inference Optimization Toolkits

**RadixAttention**


### 3. LLM Service Performance Analysis

