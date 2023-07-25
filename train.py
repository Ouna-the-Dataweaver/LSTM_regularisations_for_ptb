from pathlib import Path
import gc
from matplotlib import pyplot as plt
import numpy as np
import random
import math
import torch
import torch.optim as optim
import torch.nn as nn

from LSTM_module import Trainer

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(device)

data_path = Path("Data")

for file_path in data_path.iterdir():
    if file_path.is_file():
        file_path_str = str(file_path)
        if "test" in file_path_str:
            test_path = file_path
        elif "train" in file_path_str:
            train_path = file_path
        elif "valid" in file_path_str:
            valid_path = file_path
        else:
            words_path = file_path
# print(test_path, train_path, valid_path, words_path)


trainer = Trainer(device=device)
trainer.load_dictionary(words_path)

trainer.load_data(train_path, data_type="train")
trainer.load_data(valid_path, data_type="valid")
trainer.load_data(test_path, data_type="test")

trainer.init_model()
trainer.init_train()

trainer.train(45)
