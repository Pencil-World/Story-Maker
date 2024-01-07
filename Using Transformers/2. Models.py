# Models

"""

"""

from transformers import BertConfig, BertModel, AutoModel
import torch

# Building the config
config = BertConfig()

# Building the model from the config
model = BertModel(config)
print(config)
model = BertModel.from_pretrained("bert-base-cased")
model.save_pretrained("Models")

first = BertModel.from_pretrained("bert-base-cased")
second = AutoModel.from_pretrained("./Models/")
print(first == second)

sequences = ["Hello!", "Cool.", "Nice!"]
encoded_sequences = [
    [101, 7592, 999, 102],
    [101, 4658, 1012, 102],
    [101, 3835, 999, 102],
]
model_inputs = torch.tensor(encoded_sequences)
output = model(model_inputs)
print(output)
