# STEER-BENCH: A Benchmark for Evaluating the Steerability of Large Language Models
Paper: [STEER-BENCH: A Benchmark for Evaluating the Steerability of Large Language Models](https://arxiv.org/pdf/2505.20645)

## How to run it?
1. Install packages
```
pip install vllm
pip install transformers
pip install pandas
```
2. Run benchmark
```
python evaluation.py --config combine --model ./Qwen2.5-72B-Instruct
```

- `config`: five prompting configurations to evaluate the effectiveness of in-context learning
* `vanilla`: Vanilla (Answer multi-choice questions without context);
* `out_topic`: Out-of-topic Few-shot (Include 12 randomly sampled few-shot examples unrelated to the topic of multi-choice questions);
* `subreddit`: Subreddit Identifier (Prepend the subreddits name as context, e.g., “You are responding from r/Parenting”);
* `in_topic`: In-topic Few-shot (Include few-shot examples related to the topic of multi-choice questions);
* `combo`: In-topic Few-shot + Subreddit Identifier (Combine in-topic examples and subreddit identifier);


## Overview
Steerability, or the ability of large language models (LLMs) to adapt outputs to align with diverse community-specific norms, perspectives, and communication styles, is critical for real-world applications but remains under-evaluated. We introduce 、STEER-BENCH, a benchmark for assessing population-specific steering using contrasting Reddit communities. Covering 30 contrasting subreddit pairs across 19 domains, STEER-BENCH includes over 10,000 instruction-response pairs and validated 5,500 multiple-choice question with corresponding silver labels to test alignment with diverse community norms. Our evaluation of 13 popular LLMs using STEER-BENCH reveals that while human experts achieve an accuracy of 81% with silver labels, the best-performing models reach only around 65% accuracy depending on the domain and configuration. Some models lag behind human-level alignment by over 15 percentage points, highlighting significant gaps in community-sensitive steerability.
STEER-BENCH is a benchmark to systematically assess how effectively LLMs understand community-specific instructions, their resilience to adversarial steering attempts, and their ability to accurately represent diverse cultural and ideological perspectives.

<img src="figure/group-align-eval.png" width="600">

## STEER-BENCH Construction

To evaluate the steerability of LLMs toward specific community perspectives, we construct STEER-BENCH, a benchmark comprising automatically generated steering demonstrations and evaluation instances derived from contrasting online communities. The construction pipeline of STEER-BENCH is shown in Figure 2. We describe how we identify community pairs, collect data, generate instruction-response demonstrations $I = {(x_j, y_j)}$ for steering models, and build multiple-choice evaluation instances for assessing whether a model steered toward a community $C$ accurately reflects its views.

<img src="figure/group_align_pipeline.png" width="800">

## Evaluating LLM Steerability

Evaluation of LLM steerability using in-context learning, across 13 models using five different configurations. Models from different families demonstrate varying patterns of steerability.

<img src="figure/in_context_evaluation.png" width="700">

Domain-level steerability of LLMs using the In-topic Few-shot configuration. Accuracy is reported across 19 domains.

<img src="figure/in_context_eval_domain.png" width="600">





## Citation
```
@article{chen2025steer,
  title={STEER-BENCH: A Benchmark for Evaluating the Steerability of Large Language Models},
  author={Chen, Kai and He, Zihao and Shi, Taiwei and Lerman, Kristina},
  journal={arXiv preprint arXiv:2505.20645},
  year={2025}
}
```

Feel free to contact Kai Chen at (**kchen035@usc.edu**), if you have any questions about the paper.
