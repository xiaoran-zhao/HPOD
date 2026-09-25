# OPD-GRPO Core Project

This repository contains the core implementation of Retain OPD, Reverse OPD,
GRPO, adaptive Reverse OPD weighting, and gradient coordination. Model weights,
prompt templates, datasets, and evaluation code are not included.

## Project Structure

```text
opd_grpo_core_project/
  configs/
    current_method.yaml             # Default method configuration
  src/opd_grpo_core/
    config.py                       # Typed configuration
    losses/
      opd.py                        # Retain OPD and Reverse OPD
      grpo.py                       # GRPO and reference KL
    controllers/
      progress_grad_balance.py      # Adaptive Reverse OPD weight
    coordination/
      behavior_projection.py        # Gradient coordination
    rewards/
      unlearning.py                 # Unlearning reward construction
    training/
      batch.py                      # Strict on-policy batch state
      engine.py                     # End-to-end update orchestration
      gradients.py                  # Gradient utilities
      optimizer.py                  # Optimizer construction
      step.py                       # Coordinated update step
  pyproject.toml
  requirements.txt
```

## Training Datasets

Experiments use real-world datasets covering hazardous-knowledge removal and
copyright-related forgetting tasks.

### Unlearning Data

- **WMDP-Bio:** The biological forget corpus associated with the WMDP
  benchmark. Access is controlled by the dataset authors; see the
  [official WMDP corpora page](https://huggingface.co/datasets/cais/wmdp-corpora)
  and the [WMDP-Bio access request form](https://docs.google.com/forms/d/e/1FAIpQLSdnQc8Qn0ozSDu3VE8HLoHPvhpukX1t1dIwE5K5rJw9lnOjKw/viewform).
- **WMDP-Cyber:** The cybersecurity forget corpus is available from the
  [official WMDP corpora repository](https://huggingface.co/datasets/cais/wmdp-corpora/tree/main/cyber-forget-corpus).
- **Harry Potter:** Copyright-related forgetting follows the setup introduced
  in [*Who's Harry Potter? Approximate Unlearning in LLMs*](https://arxiv.org/abs/2310.02238).
  The copyrighted book text is not redistributed by this repository.

The WMDP benchmark code and documentation are available in the
[official WMDP repository](https://github.com/centerforaisafety/wmdp).

### Retention Data

- **WikiText-103:** General-knowledge retention data from the
  [official WikiText dataset page](https://huggingface.co/datasets/Salesforce/wikitext)
  (`wikitext-103-v1`).
