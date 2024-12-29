# [ROSE: A Reward-Oriented Data Selection Framework for LLM Task-Specific Instruction Tuning](https://arxiv.org/abs/2412.00631)

## Overview
This repository is the official implementation of ROSE [1].

## Environment Set-up

To get started with the repo, you can easily create the conda environment using the provided environment.yml file.
```setup
conda env create -n rose -f environment.yml
conda activate rose

```


## Dataset

We follow the [LESS](https://github.com/princeton-nlp/LESS?tab=readme-ov-file#less-selecting-influential-data-for-targeted-instruction-tuning) [2] to prepare four instruction-tuning datasets: Flan v2, COT, Dolly, and Open Assistant. For validation, we constructed few-shot datasets in both SFT and preference formats using SE, SHP, and SE datasets. All (train, test, validation) datasets are all included in the ./data folder.

## ROSE Data Selection
All steps are compiled into the ./run/run_rose.sh file. You can start the process using the ./run_from_start.sh script (with configuration setting).

```bash
./run/run_from_start.sh
```


## Citation
Please cite our paper if you find the repo helpful in your work:

```bibtex
@article{wu2024rose,
  title={ROSE: A Reward-Oriented Data Selection Framework for LLM Task-Specific Instruction Tuning},
  author={Wu, Yang and Zhang, Huayi and Jiao, Yizheng and Ma, Lin and Liu, Xiaozhong and Yu, Jinhong and Zhang, Dongyu and Yu, Dezhi and Xu, Wei},
  journal={arXiv preprint arXiv:2412.00631},
  year={2024}
}
```

## References
[1] Wu, Yang, Huayi Zhang, Yizheng Jiao, Lin Ma, Xiaozhong Liu, Jinhong Yu, Dongyu Zhang, Dezhi Yu, and Wei Xu. "ROSE: A Reward-Oriented Data Selection Framework for LLM Task-Specific Instruction Tuning." arXiv preprint arXiv:2412.00631 (2024).

[2] Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, and Danqi Chen. Less: Selecting influential data for targeted instruction tuning. arXiv preprint arXiv:2402.04333, 2024.

[3] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2024.
