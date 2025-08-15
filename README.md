<h1 align="center"> Learning Adaptive Parallel Reasoning <br> with Language Models </h1>

<p align="center">
  <a href="https://www.jiayipan.com/" style="text-decoration: none;">Jiayi Pan</a><sup>*</sup>,
  <a href="https://xiuyuli.com/" style="text-decoration: none;">Xiuyu Li</a><sup>*</sup>,
  <a href="https://tonylian.com/" style="text-decoration: none;">Long Lian</a><sup>*</sup>,
  <a href="https://sea-snell.github.io/" style="text-decoration: none;">Charlie Victor Snell</a>,
  <a href="https://yifeizhou02.github.io/" style="text-decoration: none;">Yifei Zhou</a>,<br>
  <a href="https://www.adamyala.org/" style="text-decoration: none;">Adam Yala</a>,
  <a href="https://people.eecs.berkeley.edu/~trevor/" style="text-decoration: none;">Trevor Darrell</a>,
  <a href="https://people.eecs.berkeley.edu/~keutzer/" style="text-decoration: none;">Kurt Keutzer</a>,
  <a href="https://www.alanesuhr.com/" style="text-decoration: none;">Alane Suhr</a>
</p>

<p align="center">
    UC Berkeley and UCSF &nbsp;&nbsp;&nbsp;<sup>*</sup> Equal Contribution
</p>

<p align="center">
<a href="https://arxiv.org/abs/2504.15466">📃 Paper</a>
•
<a href="https://github.com/Parallel-Reasoning/APR" >💻 Code</a>
•
<a href="https://huggingface.co/Parallel-Reasoning" >🤗 Data & Models</a>
</p>


![APR](./assets/apr.png)

**TL;DR**: 
We present Adaptive Parallel Reasoning (APR), a novel framework that enables language models to learn to orchestrate both serialized and parallel computations. APR trains language models to use `spawn()` and `join()` operations through end-to-end supervised training and reinforcement learning, allowing models to dynamically orchestrate their own computational workflows.
APR efficiently distributes compute, reduces latency, overcomes context window limits, and achieves state‑of‑the‑art performance on complex reasoning tasks (e.g., 83.4% vs. 60.0% accuracy at 4K context on Countdown).

> The full code will be released soon!
## Data Preparation
We provide pre-generated data on huggingface, which includes [countdown problems](https://huggingface.co/datasets/Parallel-Reasoning/countdown_problems) for training and validation, [APR SFT data](https://huggingface.co/datasets/Parallel-Reasoning/apr_sft_data), and [SoS+ SFT data](https://huggingface.co/datasets/Parallel-Reasoning/sosp_sft_data).

### Generate Data
If you want to generate the data yourself, you can run the following commands:
```bash
bash src/data/scripts/create_prob.sh
```
This will generate the countdown problems for training and validation.
```bash
bash src/data/scripts/create_search.sh
```
This will generate the APR and SoS+ search paths for training.
> [!Note]
> There are still some filtering and pre-processing steps to be done for the raw generated data for each SFT setting. We will update this part soon.

## Supervised Training
We use TPU-v3-128 for supervised training with a codebase building upon [JAX_llama](https://github.com/Sea-Snell/JAX_llama). 

Please refer to [the instructions](supervised-jax/README.md) for more details.

## Reinforcement Learning
We present TinyRL, a simple implementation of the GRPO training framework for our experiments. TinyRL is a lightweight yet performant reinforcement learning library designed to be both easy to use and extend. It integrates with [SGLang](https://github.com/sgl-project/sglang) for efficient rollout. Given the small size of the model we’re training, we haven’t implemented model parallelism, so it runs on two GPUs—one for training and one for rollout

It supports asynchronous, multi-turn, multi-agent rollouts through a general `rollout_fun` interface, with the minimal assumption that your rollout mechanism relies on calling an OpenAI-compatible API endpoint. 
```python
def rollout_fun(server_url, prefix_list, bos_token, temperature=0.5, max_workers=32):
  pass
```

Please refer to [the instructions](tinyrl/README.md) for more details.

## Evaluation

> [!IMPORTANT]
> **For evaluation, SGLang needs to be patched**.
> Remove this check in `python/sglang/srt/managers/tokenizer_manager.py` in your local SGLang repo:
> ```
> # if (
> #     obj.sampling_params.get("max_new_tokens") is not None
> #     and obj.sampling_params.get("max_new_tokens") + input_token_num
> #     >= self.context_len
> # ):
> #     raise ValueError(
> #         f"Requested token count exceeds the model's maximum context length "
> #         f"of {self.context_len} tokens. You requested a total of "
> #         f"{obj.sampling_params.get('max_new_tokens') + input_token_num} "
> #         f"tokens: {input_token_num} tokens from the input messages and "
> #         f"{obj.sampling_params.get('max_new_tokens')} tokens for the "
> #         f"completion. Please reduce the number of tokens in the input "
> #         f"messages or the completion to fit within the limit."
> #     )
> ```
> 
> This file is located at [tokenizer_manager.py](https://github.com/sgl-project/sglang/blob/45205d88a08606d5875476fbbbc76815a5107edd/python/sglang/srt/managers/tokenizer_manager.py#L350)

> [!Note]
> sgl-project/sglang#3721 introduces an `--allow-auto-truncate` option that makes this patch unnecessary. Once merged, you can use that directly.

### SoS+

The following command evaluates the SoS+ model on the validation set.
```bash
python -m src.eval.eval_sosp --ckpt <ckpt>  --temperature <temperature> --batch_size 256 --gens 1 --output_dir <output_dir> --num_gpus 8 --n_samples <n_samples> --budget <budget>
```
Where `<n_samples>` is the number of Best-of-N samples in inference, and `<budget>` is the budget for conditional generation (leave it empty if not using conditioned models). For instance, the following command evaluates the SoS+ model with 8 samples using a unconditioned checkpoint.
```bash
python -m src.eval.eval_sosp --ckpt Parallel-Reasoning/llama-sosp --temperature 1.0 --batch_size 256 --gens 1 --output_dir results/llama-sosp/ --num_gpus 8 --n_samples 8
```

### APR

First, you need to start the SGLang server for the target model. For instance:
```bash
python -m sglang.launch_server  --served-model-name model --model-path Parallel-Reasoning/llama-apr_cond10_grpo --port 2346 --dp-size 8
```

Then the following command evaluates the APR model on the validation set.
```bash
python -m src.eval.eval_apr --model_name llama-apr_cond10_grpo --ckpt Parallel-Reasoning/llama-apr_cond10_grpo --temperature 1.0 --budget 10 --use_subcall_cond
```
which evaluates the APR model with a budget of 10 child threads and uses child thread count conditioning. Do not use `--budget` and `--use_subcall_cond` for unconditioned models.


## Citation
If you find this work useful in your research, please consider citing:

```bibtex
@article{pan2025learning,
  title   = {Learning Adaptive Parallel Reasoning with Language Models},
  author  = {Jiayi Pan and Xiuyu Li and Long Lian and Charlie Snell and Yifei Zhou and Adam Yala and Trevor Darrell and Kurt Keutzer and Alane Suhr},
  year    = {2025},
  journal = {arXiv preprint arXiv: 2504.15466}
}
```
