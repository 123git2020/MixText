# MixText
This repo contains codes for the following paper: 

*Jiaao Chen, Zichao Yang, Diyi Yang*: [MixText: Linguistically-Informed Interpolation of Hidden Space for Semi-Supervised Text Classification](https://aclanthology.org/2020.acl-main.194/). In Proceedings of the 58th Annual Meeting of the Association of Computational Linguistics (ACL'2020)

If you would like to refer to it, please cite the paper mentioned above. 


## Getting Started
These instructions will get you running the codes of MixText.

### Requirements
* transformers
* Pandas, Numpy, Pickle


### Code Structure
```
|__ data/
        |__ AG_News/ --> Datasets for AG News
            |__ Preprocess.ipynb --> Jupyter Notebook for process/clean the dataset
            |__ back_translate.ipynb --> Jupyter Notebook for back translating the dataset
            |__ classes.txt --> Classes for AG_News dataset
            |__ train.csv --> Original training dataset
            |__ test.csv --> Original testing dataset
            |__ train_clean.csv --> Cleaned training dataset
            |__ test_clean.csv --> Cleaned testing dataset
            |__ zh.pkl --> Training dataset translated to Chinese
            |__ zh_back.pkl --> Back translated training dataset with Chinese as middle language

        |__ yahoo_answers/ --> Datasets for yahoo_answers


|__code/
        |__ read_data.py --> Codes for reading the dataset; forming labeled training set, unlabeled training set, development set and testing set; building dataloaders
        |__ normal_bert.py --> Codes for BERT baseline model
        |__ normal_train.py --> Codes for training BERT baseline model
        |__ mixtext.py --> Codes for our proposed TMix/MixText model
        |__ train.py --> Codes for training/testing TMix/MixText 
```

### Downloading the data
Please download the dataset and put them in the data folder. You can find Yahoo Answers, AG News, DB Pedia [here](https://github.com/LC-John/Yahoo-Answers-Topic-Classification-Dataset), IMDB [here](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

### Pre-processing the data

_It seems that there is no yahoo dataset identical to the one used by the authors._

Note that for AG News and DB Pedia, we only utilize the content (without titles) to do the classifications, and for IMDB we do not perform any pre-processing.

We utilize [pygtrans](https://pypi.org/project/pygtrans/) which is a translation library using google translate to perform back translation on the training dataset. Please refer to `./data/AG_News/back_translate.ipynb` for details.

Here, we have put two examples of back translated data, `zh_back.pkl`, in `./data/AG_News/` as well. You can directly use them for Yahoo Answers or generate your own back translated data followed the `./data/AG_News/back_translate.ipynb`.



### Training models
These section contains instructions for training models on AG News using 4 labeled data per class for training.


#### Training BERT baseline model
Please run `./code/normal_train.py` to train the BERT baseline model (only use labeled training data):
```
python ./code/normal_train.py --gpu 0,1 --n-labeled 10 --data-path ./data/AG_News/ \
--batch-size 8 --epochs 20 
```

#### Training TMix model
Please run `./code/train.py` to train the TMix model (only use labeled training data):
```
python ./code/train.py --gpu 0,1 --n-labeled 10 --data-path ./data/AG_News/ \
--batch-size 8 --batch-size-u 1 --epochs 50 --val-iteration 20 \
--lambda-u 0 --T 0.5 --alpha 16 --mix-layers-set 7 9 12 --separate-mix True 
```


#### Training MixText model
Please run `./code/train.py` to train the MixText model (use both labeled and unlabeled training data):
```
python ./code/train.py --gpu 0,1,2,3 --n-labeled 10 \
--data-path ./data/AG_News/ --batch-size 4 --batch-size-u 8 --epochs 20 --val-iteration 1000 \
--lambda-u 1 --T 0.5 --alpha 16 --mix-layers-set 7 9 12 \
--lrmain 0.000005 --lrlast 0.0005
```

### Results
The [runs](https://github.com/123git2020/MixText/tree/master/runs) folder gives tensorboard logs of two training runs with different labeld data.
