# Γ-Trans
Implementaion of Exploiting Labeled and Unlabeled Data via Transformer Fine-tuning for Peer-Review Score Prediction

## Environment Installation
1. Download and install [Anaconda](https://www.anaconda.com/products/individual)
2. Create environment
```
conda env create -f environment.yml
conda activate gamma_trans
```

## Dataset
1. [PeerRead](https://github.com/allenai/PeerRead)
2. [ScisummNet](https://cs.stanford.edu/~myasu/projects/scisumm_net/)
```
cd datasets
./download.sh
```

## Γ-Trans
```
pthon run_gamma_trans.py --aspects {recommend, substance, appropriate, comparison, soundness, originality, clarity, impact}
```

## Ladder
```
pthon run_ladder.py --aspects {recommend, substance, appropriate, comparison, soundness, originality, clarity, impact}
```

## Γ-model
```
pthon run_gamma_model.py --aspects {recommend, substance, appropriate, comparison, soundness, originality, clarity, impact}
```
