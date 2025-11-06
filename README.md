## TranSMOTE: Transformer Based Synthetic Minority Oversampling Technique
Priyobrata Mondal, Soumi Pal, Swagatam Das


_________________

This is the official implementation of TranSMOTE in the paper [TranSMOTE: Transformer Based Synthetic Minority Oversampling Technique (ICVGIP 2025)].

### Installation and data pre-processing
```bash
pip install -r requirements.txt
```
- Create a folder for saved models
- Create a folder results
- Create an imbalanced SVHN and save the training images and labels in data/svhn/train_x.pt and data/svhn/train_y.pt respectively.
```bash
cd data
python3 validation_save.py
cd ..
```



### Training 
```bash
python3 training.py
```

### Generation
python3 generation.py

### Classification
sh run.sh


# TranSMOTE
