# Behind the pipeline

"""
Operation: Tokenizer --> Model (Transformer --> Head) --> Post Processing
Flow: Model Input (Raw Text) --> Input IDs --> Hidden States --> Logits --> Model Output (Predictions)
Together: Model Input (Raw Text) --> Tokenizer --> Input IDs --> Transformer --> Hidden States --> Head --> Logits --> Post Processing --> Model Output (Predictions)
"""

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)

model = AutoModel.from_pretrained(checkpoint) # default model using the Transformer checkpoint which does not contain a head layer
outputs = model(**inputs) # outputs the Transformer outputs (hidden states or features) that give contextual understanding of that input by the Transformer model
print(outputs.last_hidden_state.shape)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint) # sequence classification head model (SoftMax) using the Transformer checkpoint
outputs = model(**inputs) # outputs the classification of the sentences as positive or negative after processing Transformer output through the model head
print(outputs.logits.shape)

print(outputs.logits) # logits: raw, unnormalized scores outputted by the last layer of the model
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1) # post processing
print(predictions)
print(model.config.id2label)
