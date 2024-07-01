# "What do you want from theory alone?" Experimenting with Tight Auditing of Differentially Private Synthetic Data Generation
This repository contains the source code for the paper _"What do you want from theory alone?" Experimenting with Tight Auditing of Differentially Private Synthetic Data Generation_ by M.S.M.S. Annamalai, G. Ganev, E. De Cristofaro, to appear at [USENIX Security 2024](https://www.usenix.org/conference/usenixsecurity24).

## Install
### Manual
Dependencies are managed by `conda`.  
1. The required dependencies can be installed using the command `conda env create -f env.yml` and then run `conda activate synth-audit`.  
2. Additionally, install the (modified) versions of the synthetic data generation libraries by running `libs/install.sh`.

### Docker
For simplicity, we have published a docker image at `msundarmsa/synth-audit:1.0` with the dependencies pre-installed.  
1. Pull and run the image using the command `docker run -it msundarmsa/synth-audit:1.0 /bin/bash`.
2. Then cd into the folder `cd ~/synth-audit/audit` and activate the environment using `conda activate synth-audit`.

## Run
All commands are run from inside the `audit` folder.
### 1. Prepare synthetic data
Synthetic data can be fit and generated from the raw datasets by running `python3 prep_synths.py`.  
E.g., `python3 prep_synths.py --data_name adult --neighbour edit --target_idx 61 --n_synth 1000 --n_reps 100 --model DPartPB --epsilon 10.0 --n_procs 32 --out_dir exp_data/test/` prepares appropriate $D$ and $D^-$ and generates 100 synthetic datasets (50 from $D$ and 50 from $D^-$) using PrivBayes (Hazy).

### 2. Run attack
The various attacks can be run using `python3 run_attack.py`.  
E.g., `python3 run_attack.py --data_name adult --neighbour edit --target_idx 61 --n_shadow 60 --n_valid 20 --n_test 20 --model DPartPB --epsilon 10.0 --out_dir exp_data/test/ --attack_type bb_querybased` runs the Black-box (Querybased) attack on the generated synthetic datasets.

### 3. Experiments
We provide the exact scripts we use to run experiments under the `scripts/` folder, which should have more options that you can play around with.  
Results can then be generated using the `analyze_results.ipynb` notebook.  
Lastly, results can be plotted using `plot_results.ipynb` notebook.

## Notes
For simplicity, we have renamed the models within the python library as follows:  
| Model | Description |  
|-------|-------------|  
| DPartPB | PrivBayes (Hazy) |  
| DSynthPB | PrivBayes (DataSynthesizer) |  
| NIST_MST | MST (NIST) |  
| MST | MST (Smartnoise) |  
| DPWGAN | DPWGAN (NIST) |  
| DPWGANCity | DPWGAN (SynthCity) |  
| DSynthPB_v014 | PrivBayes (DataSynthesizer v0.1.4) |  
