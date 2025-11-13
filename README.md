# ROSE: A Reward-Oriented Data Selection Framework for LLM Task-Specific Instruction Tuning

[![Paper](https://img.shields.io/badge/Paper-EMNLP%202025-blue)](https://aclanthology.org/2025.findings-emnlp.710/)
[![PDF](https://img.shields.io/badge/PDF-Download-red)](https://aclanthology.org/2025.findings-emnlp.710.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-2412.00631-b31b1b.svg)](https://arxiv.org/abs/2412.00631)

**Official implementation of "ROSE: A Reward-Oriented Data Selection Framework for LLM Task-Specific Instruction Tuning"**

*Yang Wu, Huayi Zhang, Yizheng Jiao, Lin Ma, Xiaozhong Liu, Jinhong Yu, Dongyu Zhang, Dezhi Yu, Wei Xu*

**Published in:** Findings of the Association for Computational Linguistics: EMNLP 2025

---

## Overview

Instruction tuning has demonstrated the significant potential of large language models (LLMs) in producing more human-controllable and effective outputs across various domains. However, a critical challenge remains: **selecting the most effective training data for task-specific instruction tuning**.

**ROSE** introduces a novel reward-oriented data selection framework that addresses a fundamental issue in current methods: the misalignment between instruction tuning loss (cross-entropy loss) and actual task performance. By leveraging **pairwise preference loss as a reward signal**, ROSE optimizes data selection to truly improve target task performance rather than merely minimizing training loss.

### Key Features

- **Reward-Oriented Selection**: Uses preference-based reward signals instead of traditional cross-entropy loss for data selection
- **Influence Function Framework**: Adapts influence formulation to approximate the impact of training data on few-shot preference validation sets
- **High Efficiency**: Achieves competitive results with only **5% of training data** compared to full dataset fine-tuning
- **Strong Generalizability**: Robust performance across multiple benchmark datasets and diverse model architectures
- **State-of-the-Art**: Surpasses existing data selection methods for task-specific instruction tuning

### Why ROSE?

Traditional data selection methods rely on similarity metrics to align training data with test data distribution, aiming to minimize instruction tuning loss. However, instruction tuning loss often fails to exhibit a monotonic relationship with actual task performance. ROSE solves this by:

1. Using **DPO (Direct Preference Optimization)** gradients instead of SFT (Supervised Fine-Tuning) gradients
2. Computing influence scores relative to **preference validation sets**
3. Selecting data that **directly improves reward model performance** on target tasks

---

## Environment Setup

Create the conda environment using the provided configuration file:

```bash
conda env create -n rose -f environment.yml
conda activate rose
pip install fast-jl==0.1.3
```

### Requirements

- Python 3.10+
- PyTorch 2.1.2
- Transformers 4.36.2
- CUDA-capable GPU(s)
- See `environment.yml` for complete dependencies

---

## Dataset

ROSE follows the [LESS](https://github.com/princeton-nlp/LESS) framework to prepare instruction-tuning datasets:

### Training Data
- **Flan v2**: Large-scale instruction tuning dataset
- **COT (Chain-of-Thought)**: Reasoning-focused instructions
- **Dolly**: High-quality human-generated instruction-response pairs
- **Open Assistant**: Community-contributed dialogue data

All training data should be placed in `./data/train/processed/` with the following structure:
```
data/train/processed/
├── dolly/dolly_data.jsonl
├── cot/cot_data.jsonl
├── flan_v2/flan_v2_data.jsonl
└── oasst1/oasst1_data.jsonl
```

### Validation Data
Few-shot datasets in both **SFT** and **preference formats**:
- **SE (Stack Exchange)**: Technical Q&A from Stack Exchange
- **SHP (Stanford Human Preferences)**: Reddit-based preference data
- **HH (Helpful & Harmless)**: Anthropic's dialogue preferences

Validation data is organized by task and shot number in `./data/validation/`.

### Test Data
Task-specific test sets for evaluation in `./data/test/`.

**Note**: All datasets (train, validation, test) are included in the `./data/` directory.

---

## Quick Start

### Configuration

Edit the configuration in `./run/run_from_start.sh`:

```bash
data_seed=888                              # Experiment seed for reproducibility
percentage=0.05                             # Data selection percentage (5%)
model_load_path=meta-llama/Meta-Llama-3.1-8B  # Base LLM path
devices="0 1 2 3"                          # Available CUDA devices
max_collect_samples=None                    # Training samples limit (None for all)
projection_dims=8192                        # Gradient projection dimension
```

### Run ROSE Pipeline

Execute the complete ROSE pipeline:

```bash
cd run
./run_from_start.sh
```

This script orchestrates all six steps of the ROSE framework automatically.

---

## ROSE Pipeline

The ROSE framework consists of six main steps:

### Step 1: Warmup with LoRA
```bash
./run/step1_warmup_lora.sh
```
Fine-tune the base model with LoRA adapters on a small portion of training data to create intermediate checkpoints.

**Output**: Checkpoints saved in `./out/{model_name}-p{percentage}-warmup-lora-seed{seed}/`

### Step 2: Compute Training Gradients
```bash
./run/step2_training_storage.sh
```
Compute and store gradients for all training data at each checkpoint using Adam optimizer state.

**Output**: Training gradients in `./grad/{model_name}-p{percentage}-warmup-lora-seed{seed}/{dataset}-ckpt{num}-adam/`

### Step 3: Compute Validation Gradients
```bash
./run/step3_validate_storage.sh
```
Compute DPO (preference) gradients for few-shot validation sets at each checkpoint.

**Output**: Validation gradients in `./grad/{model_name}-p{percentage}-warmup-lora-seed{seed}/{task}-ckpt{num}-dpo-sgd/`

### Step 4: Calculate Influence Scores
```bash
./run/step4_influence_calculation.sh
```
Compute influence scores by projecting training gradients onto validation gradients using the influence function framework.

**Formula**: 
```
influence(z_train) ≈ ∇L_DPO(z_val) · ∇L_SFT(z_train)
```

**Output**: Influence scores in `./rose_result/rose_dpo_data/`

### Step 5: Save Selected Data
```bash
./run/step5_save_selected_data.sh
```
Select top-k most influential training samples based on influence scores.

**Output**: Selected data in `./rose_result/rose_dpo_data/{task}/top_p{percentage}.jsonl`

### Step 6: Fine-tune on Selected Data
```bash
./run/step6_finetune.sh
```
Fine-tune the base model on the selected high-quality data.

**Output**: Final model in `./out/{model_name}-rose-dpo-{task}-p{percentage}-lora-seed{seed}/`

---

## Project Structure

```
ROSE/
├── data/                           # All datasets
│   ├── train/                      # Training data
│   │   └── processed/              # Preprocessed training datasets
│   │       ├── dolly/
│   │       ├── cot/
│   │       ├── flan_v2/
│   │       └── oasst1/
│   ├── validation/                 # Few-shot validation sets
│   │   ├── hh/                     # HH validation (1-50 shot)
│   │   ├── se/                     # SE validation (1-50 shot)
│   │   └── shp/                    # SHP validation (1-50 shot)
│   └── test/                       # Test datasets
│       ├── hh/
│       ├── se/
│       └── shp/
├── rose/                           # Core ROSE implementation
│   ├── arguments.py                # Training arguments configuration
│   ├── build_storage.py            # Gradient storage builder
│   ├── data_formatter.py           # Data preprocessing utilities
│   ├── data_selection_save.py      # Save selected data
│   ├── dpo_gradient.py             # DPO gradient computation
│   ├── get_learning_rates.py       # Extract learning rates from checkpoints
│   ├── influence_estimation.py     # Influence function calculation
│   ├── meta_data_collection.py     # Metadata collection
│   ├── model_training.py           # Model training utilities
│   ├── preference_dataset.py       # Preference dataset handler
│   └── utils.py                    # General utilities
├── run/                            # Execution scripts
│   ├── run_from_start.sh           # Main entry point
│   ├── run_rose.sh                 # ROSE pipeline orchestrator
│   ├── step1_warmup_lora.sh        # Step 1: Warmup
│   ├── step2_training_storage.sh   # Step 2: Training gradients
│   ├── step3_validate_storage.sh   # Step 3: Validation gradients
│   ├── step4_influence_calculation.sh  # Step 4: Influence scores
│   ├── step5_save_selected_data.sh # Step 5: Data selection
│   └── step6_finetune.sh           # Step 6: Final fine-tuning
├── environment.yml                 # Conda environment specification
└── README.md
```

---

## Results

ROSE demonstrates significant improvements in data efficiency and model performance:

### Data Efficiency
- **5% of training data** achieves competitive performance with full-dataset fine-tuning
- Outperforms random selection, embedding-based selection, and gradient-based methods
- Scales effectively across different data budgets (1%-20%)

### Task Performance
Evaluated on three benchmark categories:
- **SE (Stack Exchange)**: 10 domain-specific QA tasks
- **SHP (Stanford Human Preferences)**: 18 Reddit-based preference tasks
- **HH (Helpful & Harmless)**: Safety-aligned dialogue generation

### Key Findings
1. **Preference signals > Cross-entropy loss**: Using DPO gradients significantly outperforms SFT gradients
2. **Robust across architectures**: Validated on LLaMA-3.1, Mistral, and other model families
3. **Few-shot generalization**: Works effectively with as few as 1-5 validation examples
4. **Computational efficiency**: Gradient computation is parallelizable and cached for reuse

See the paper for detailed experimental results and analysis.

---

## Advanced Usage

### Custom Base Models

To use a different base model, update `model_load_path` in `run_from_start.sh`:

```bash
model_load_path="your-org/your-model"  # e.g., mistralai/Mistral-7B-v0.1
```

### Adjust Data Selection Percentage

Modify the `percentage` parameter:

```bash
percentage=0.10  # Select 10% of training data
```

### Custom Validation Sets

Add your own validation data in the appropriate format:
- Preference format: `{"prompt": "...", "chosen": "...", "rejected": "..."}`
- SFT format: `{"prompt": "...", "completion": "..."}`

Place files in `./data/validation/your_task/` and update the task list in `run_rose.sh`.

### Multi-GPU Training

Specify available GPUs:

```bash
devices="0 1 2 3 4 5 6 7"  # Use 8 GPUs
```

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{wu-etal-2025-rose,
    title = "{ROSE}: A Reward-Oriented Data Selection Framework for {LLM} Task-Specific Instruction Tuning",
    author = "Wu, Yang  and
      Zhang, Huayi  and
      Jiao, Yizheng  and
      Ma, Lin  and
      Liu, Xiaozhong  and
      Yu, Jinhong  and
      Zhang, Dongyu  and
      Yu, Dezhi  and
      Xu, Wei",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2025",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-emnlp.710/",
    doi = "10.18653/v1/2025.findings-emnlp.710",
    pages = "13200--13219",
    ISBN = "979-8-89176-335-7",
    abstract = "Instruction tuning has underscored the significant potential of large language models (LLMs) in producing more human controllable and effective outputs in various domains. In this work, we focus on the data selection problem for task-specific instruction tuning of LLMs. Prevailing methods primarily rely on the crafted similarity metrics to select training data that aligns with the test data distribution. The goal is to minimize instruction tuning loss on the test data, ultimately improving performance on the target task. However, it has been widely observed that instruction tuning loss (i.e., cross-entropy loss for next token prediction) in LLMs often fails to exhibit a monotonic relationship with actual task performance. This misalignment undermines the effectiveness of current data selection methods for task-specific instruction tuning. To address this issue, we introduce ROSE, a novel Reward-Oriented inStruction data sElection method which leverages pairwise preference loss as a reward signal to optimize data selection for task-specific instruction tuning. Specifically, ROSE adapts an influence formulation to approximate the influence of training data points relative to a few-shot preference validation set to select the most task-related training data points. Experimental results show that by selecting just 5{\%} of the training data using ROSE, our approach can achieve competitive results compared to fine-tuning with the full training dataset, and it surpasses other state-of-the-art data selection methods for task-specific instruction tuning. Our qualitative analysis further confirms the robust generalizability of our method across multiple benchmark datasets and diverse model architectures."
}
```

**arXiv preprint**:
```bibtex
@article{wu2024rose,
  title={ROSE: A Reward-Oriented Data Selection Framework for LLM Task-Specific Instruction Tuning},
  author={Wu, Yang and Zhang, Huayi and Jiao, Yizheng and Ma, Lin and Liu, Xiaozhong and Yu, Jinhong and Zhang, Dongyu and Yu, Dezhi and Xu, Wei},
  journal={arXiv preprint arXiv:2412.00631},
  year={2024}
}
```

---

## Related Work

ROSE builds upon and extends several key research directions:

### Influence Functions for Data Selection
- **LESS** [Xia et al., 2024]: Influential data selection using gradient-based influence estimation
- **TracIn** [Pruthi et al., 2020]: Tracing model predictions to training data

### Preference-Based Learning
- **DPO** [Rafailov et al., 2024]: Direct Preference Optimization without explicit reward modeling
- **RLHF** [Ouyang et al., 2022]: Reinforcement Learning from Human Feedback

### Instruction Tuning
- **Flan** [Wei et al., 2022]: Fine-tuned Language Models are instruction-following
- **Self-Instruct** [Wang et al., 2023]: Aligning language models with self-generated instructions

---

## References

[1] Wu, Yang, Huayi Zhang, Yizheng Jiao, Lin Ma, Xiaozhong Liu, Jinhong Yu, Dongyu Zhang, Dezhi Yu, and Wei Xu. "ROSE: A Reward-Oriented Data Selection Framework for LLM Task-Specific Instruction Tuning." *Findings of EMNLP 2025*.

[2] Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, and Danqi Chen. "LESS: Selecting Influential Data for Targeted Instruction Tuning." *arXiv preprint arXiv:2402.04333*, 2024.

[3] Rafailov, Rafael, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea Finn. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." *Advances in Neural Information Processing Systems 36*, 2024.

---

## Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact the authors via the paper correspondence

**Paper**: [https://aclanthology.org/2025.findings-emnlp.710/](https://aclanthology.org/2025.findings-emnlp.710/)  
**arXiv**: [https://arxiv.org/abs/2412.00631](https://arxiv.org/abs/2412.00631)  
**PDF**: [https://aclanthology.org/2025.findings-emnlp.710.pdf](https://aclanthology.org/2025.findings-emnlp.710.pdf)

---

## License

This project is released for research purposes. Please cite our paper if you use this code or methodology in your research.

---

## Acknowledgments

We thank the authors of LESS for their foundational work on gradient-based data selection, and the creators of the DPO framework for insights into preference-based optimization. We also acknowledge the datasets used in our evaluation: Stack Exchange, Stanford Human Preferences, and Anthropic's HH-RLHF.

---
