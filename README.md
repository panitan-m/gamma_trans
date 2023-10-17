# Γ-Trans
Implementation of [Exploiting Labeled and Unlabeled Data via Transformer Fine-tuning for Peer-Review Score Prediction](https://aclanthology.org/2022.findings-emnlp.164.pdf)

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
python run_gamma_trans.py --aspects {recommend, substance, comparison, soundness, originality, clarity, impact}
```

## Ladder
```
python run_ladder.py --aspects {recommend, substance, comparison, soundness, originality, clarity, impact}
```

## Γ-model
```
python run_gamma_model.py --aspects {recommend, substance, comparison, soundness, originality, clarity, impact}
```
## Citation    
    @inproceedings{muangkammuen-etal-2022-exploiting,
      title = "Exploiting Labeled and Unlabeled Data via Transformer Fine-tuning for Peer-Review Score Prediction",
      author = "Muangkammuen, Panitan and Fukumoto, Fumiyo and Li, Jiyi and Suzuki, Yoshimi",
      booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
      month = dec,
      year = "2022",
      address = "Abu Dhabi, United Arab Emirates",
      publisher = "Association for Computational Linguistics",
      url = "https://aclanthology.org/2022.findings-emnlp.164",
      doi = "10.18653/v1/2022.findings-emnlp.164",
      pages = "2233--2240",
    }
