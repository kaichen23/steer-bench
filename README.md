# STEER-BENCH: A Benchmark for Evaluating the \\Steerability of Large Language Models
Paper: 

## Overview
Steerability, or the ability of large language models (LLMs) to adapt outputs to align with diverse community-specific norms, perspectives, and communication styles, is critical for real-world applications but remains under-evaluated. We introduce 、\textbf{STEER-BENCH}, a benchmark for assessing population-specific steering using contrasting Reddit communities. Covering 30 contrasting subreddit pairs across 19 domains, STEER-BENCH includes over 10,000 instruction-response pairs and validated 5,500 multiple-choice question with corresponding silver labels to test alignment with diverse community norms. Our evaluation of 13 popular LLMs using STEER-BENCH reveals that while human experts achieve an accuracy of 81% with silver labels, the best-performing models reach only around 65% accuracy depending on the domain and configuration. Some models lag behind human-level alignment by over 15 percentage points, highlighting significant gaps in community-sensitive steerability.
STEER-BENCH is a benchmark to systematically assess how effectively LLMs understand community-specific instructions, their resilience to adversarial steering attempts, and their ability to accurately represent diverse cultural and ideological perspectives.

<img src="figure/manipulate_pipeline.png" width="800">

## IdeoINST: A Collection of Ideologically Driven Instructional Data

we create a dataset named \textsc{IdeoINST} for \textbf{ideo}logically-charged \textbf{inst}ruction tuning. \textsc{IdeoINST} comprises of around 6,000 opinion-eliciting instructions across six sociopolitical topics, each paired with dual responses---one reflecting a left-leaning bias and one reflecting a right-leaning bias.

<img src="figure/data_generation_pipeline.png" width="800">











## Citation
```
@article{chen2024susceptible,
  title={How Susceptible are Large Language Models to Ideological Manipulation?},
  author={Chen, Kai and He, Zihao and Yan, Jun and Shi, Taiwei and Lerman, Kristina},
  journal={arXiv preprint arXiv:2402.11725},
  year={2024}
}
```

Feel free to contact Kai Chen at (**kchen035@usc.edu**), if you have any questions about the paper.
