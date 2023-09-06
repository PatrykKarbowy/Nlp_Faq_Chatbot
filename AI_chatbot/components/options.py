import argparse

import consts.dl_model_training
import consts.directories
from components.config import DLTrainingConfig

def parse_arg() -> DLTrainingConfig:
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-batch_size", "--batch_size", type=int, default=consts.dl_model_training.DEFAULT_BATCH_SIZE)
    parser.add_argument("-learning_rate", "--learning_rate", type=float, default=consts.dl_model_training.DEFAULT_LEARNING_RATE)
    parser.add_argument("-max_epochs", "--max_epochs", type=int, default=consts.dl_model_training.DEFAULT_MAX_EPOCHS)
    parser.add_argument("-hidden_size", "--hidden_size", type=str, default=consts.dl_model_training.DEFAULT_HIDDEN_SIZE)
    parser.add_argument("-optimizer", "--optimizer", type=str, default=consts.dl_model_training.DEFAULT_OPTIMIZER)
    parser.add_argument("-loss", "--loss", type=str, default=consts.dl_model_training.DEFAULT_LOSS)
    parser.add_argument("-device", "--device", type=str, default=consts.dl_model_training.DEFAULT_DEVICE)
    parser.add_argument("-ignore_words", "--ignore_words", type=str, default=consts.dl_model_training.DEFAULT_IGNORE_WORDS)
    
    parser.add_argument("-data_dir", "--data_dir", type=str, default=consts.directories.DATA_DIR)
    
    args = parser.parse_args()
    cfg = DLTrainingConfig(**vars(args))
    return cfg