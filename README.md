# ALDI
ALDI is a state-of-the-art framework for cold-start recommendation. It addresses the three difference (i.e., rating distribution difference, ranking difference and identification difference) between the warm model and cold model in a general knowledge distillation-based framework.

# Quick Start
- Run `python main.py --embed_meth bprmf --dataset CiteULike --model ALDI`.

# From Scratch
1. Pre-process the dataset
    - Go the `data` directory by `cd data/`.
    - Split the dataset by `python split.py --dataset CiteULike`.
    - Formulate the data and by `python convert.py --dataset CiteULike`. The processed results will be stored in `$root_path/data/$dataset_name/`

2. Pre-train warm model.
    - Go back to the root directory. 
    - Go to the the directory of warm model by `cd warm_model/`.
    - Pre-train the warm model Matrix Factorization by running `python bprmf.py --dataset CiteULike`. The trained embeddings will be also stored in `$root_path/data/$dataset_name/`

3. Train and evaluate cold model.
    - Go back to the root directory.
    - Train and evaluate ALDI by running `python main.py --embed_meth bprmf --dataset CiteULike --model ALDI`. 
