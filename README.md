# QA-DMN-PyTorch
This is an implementation of the Dynamic Memory Network proposed in [Ask Me Anything: Dynamic Memory Networks for Natural Language Processing](http://proceedings.mlr.press/v48/kumar16.pdf) and [Dynamic Memory Networks for Visual and Textual Question Answering](http://proceedings.mlr.press/v48/xiong16.pdf) for passage re-ranking in QA.

## Requirements
This code is tested with Python 3.8.10 and
* torch==1.8.1
* pytorch-lightning==1.3.3
* h5py==3.2.1
* numpy==1.20.2
* tqdm==4.61.0
* torchtext==0.9.1
* nltk==3.6.2

## Cloning
Clone this repository using `git clone --recursive` to get the submodule.

## Usage
The following datasets are currently supported:
* [ANTIQUE](https://ciir.cs.umass.edu/downloads/Antique/)
* [FiQA Task 2](https://sites.google.com/view/fiqa/home)
* [InsuranceQA V2](https://github.com/shuzi/insuranceQA)
* [TREC-DL 2019](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019)
* Any dataset in generic TREC format

### Preprocessing
First, preprocess your dataset:
```
usage: preprocess.py [-h] [--num_negatives NUM_NEGATIVES]
                     [--pw_num_negatives PW_NUM_NEGATIVES]
                     [--pw_query_limit PW_QUERY_LIMIT] [--random_seed RANDOM_SEED]
                     SAVE
                     {antique,fiqa,insuranceqa,trecdl2019passage,trecdl2019document,trec}
                     ...

positional arguments:
  SAVE                  Where to save the results
  {antique,fiqa,insuranceqa,trecdl2019passage,trecdl2019document,trec}
                        Choose a dataset

optional arguments:
  -h, --help            show this help message and exit
  --num_negatives NUM_NEGATIVES
                        Number of negatives per positive (pointwise training)
                        (default: 1)
  --pw_num_negatives PW_NUM_NEGATIVES
                        Number of negatives per positive (pairwise training)
                        (default: 16)
  --pw_query_limit PW_QUERY_LIMIT
                        Maximum number of training examples per query (pairwise
                        training) (default: 64)
  --random_seed RANDOM_SEED
```

Next, create a vocabulary:
```
usage: create_vocab.py [-h] [--max_size MAX_SIZE] [--cache CACHE] [--vectors VECTORS]
                       [--out_file OUT_FILE] [--random_seed RANDOM_SEED]
                       DATA_FILE

positional arguments:
  DATA_FILE             File that holds the queries and documents

optional arguments:
  -h, --help            show this help message and exit
  --max_size MAX_SIZE   Maximum vocabulary size (default: None)
  --cache CACHE         Torchtext cache (default: None)
  --vectors VECTORS     Pre-trained vectors (default: glove.840B.300d)
  --out_file OUT_FILE   Where to save the vocabulary (default: vocab.pkl)
  --random_seed RANDOM_SEED
                        Random seed (default: 123)
```

### Training and Evaluation
Use the training script to train a new model and save checkpoints:
```
usage: train.py [-h] [--accumulate_grad_batches ACCUMULATE_GRAD_BATCHES]
                [--max_epochs MAX_EPOCHS] [--gpus GPUS [GPUS ...]]
                [--val_check_interval VAL_CHECK_INTERVAL] [--save_top_k SAVE_TOP_K]
                [--limit_val_batches LIMIT_VAL_BATCHES]
                [--limit_train_batches LIMIT_TRAIN_BATCHES]
                [--limit_test_batches LIMIT_TEST_BATCHES] [--precision {16,32}]
                [--accelerator ACCELERATOR] [--rep_dim REP_DIM]
                [--attention_dim ATTENTION_DIM] [--agru_dim AGRU_DIM]
                [--num_episodes NUM_EPISODES] [--dropout DROPOUT] [--lr LR]
                [--loss_margin LOSS_MARGIN] [--batch_size BATCH_SIZE]
                [--training_mode {pointwise,pairwise}] [--val_patience VAL_PATIENCE]
                [--save_dir SAVE_DIR] [--random_seed RANDOM_SEED]
                [--load_weights LOAD_WEIGHTS] [--test]
                DATA_DIR FOLD_NAME VOCAB

positional arguments:
  DATA_DIR              Folder with all preprocessed files
  FOLD_NAME             Name of the fold (within DATA_DIR)
  VOCAB                 Vocabulary file

optional arguments:
  -h, --help            show this help message and exit
  --accumulate_grad_batches ACCUMULATE_GRAD_BATCHES
                        Update weights after this many batches (default: 1)
  --max_epochs MAX_EPOCHS
                        Maximum number of epochs (default: 20)
  --gpus GPUS [GPUS ...]
                        GPU IDs to train on (default: None)
  --val_check_interval VAL_CHECK_INTERVAL
                        Validation check interval (default: 1.0)
  --save_top_k SAVE_TOP_K
                        Save top-k checkpoints (default: 1)
  --limit_val_batches LIMIT_VAL_BATCHES
                        Use a subset of validation data (default: 9223372036854775807)
  --limit_train_batches LIMIT_TRAIN_BATCHES
                        Use a subset of training data (default: 9223372036854775807)
  --limit_test_batches LIMIT_TEST_BATCHES
                        Use a subset of test data (default: 9223372036854775807)
  --precision {16,32}   Floating point precision (default: 32)
  --accelerator ACCELERATOR
                        Distributed backend (accelerator) (default: ddp)
  --rep_dim REP_DIM     The dimension of fact and query representations and memory
                        (default: 256)
  --attention_dim ATTENTION_DIM
                        The dimension of the linear layer applied to the interactions
                        (default: 256)
  --agru_dim AGRU_DIM   The hidden dimension of the attention GRU (default: 256)
  --num_episodes NUM_EPISODES
                        The number of episodes (default: 4)
  --dropout DROPOUT     Dropout percentage (default: 0.1)
  --lr LR               Learning rate (default: 0.001)
  --loss_margin LOSS_MARGIN
                        Margin for pairwise loss (default: 0.2)
  --batch_size BATCH_SIZE
                        Batch size (default: 32)
  --training_mode {pointwise,pairwise}
                        Training mode (default: pairwise)
  --val_patience VAL_PATIENCE
                        Validation patience (default: 3)
  --save_dir SAVE_DIR   Directory for logs, checkpoints and predictions (default: out)
  --random_seed RANDOM_SEED
                        Random seed (default: 123)
  --load_weights LOAD_WEIGHTS
                        Load pre-trained weights before training (default: None)
  --test                Test the model after training (default: False)
```
Use the `--test` argument to run the model on the testset using the best checkpoint after training. This will create output files (one per GPU) in your experiment directory. You can then use `evaluate.py` to create a TREC runfile that can be evaluated with the TREC evaluation tool.
