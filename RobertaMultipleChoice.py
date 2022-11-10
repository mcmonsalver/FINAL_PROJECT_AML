from transformers import RobertaTokenizer, RobertaForMultipleChoice
import torch
import json
import jsonlines
from torch import nn
from transformers import TrainingArguments
from transformers import DefaultDataCollator
from transformers import Trainer
from datasets import load_dataset
from tqdm import tqdm
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch


tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
model = RobertaForMultipleChoice.from_pretrained("roberta-large")

"""
##### CARGAR ARCHIVOS TRAIN Y VALID #####

with open('hellaswag/data/hellaswag_train.jsonl') as json_file:
    json_list_train = list(json_file)

list_train = []
for json_str in json_list_train:
    result = json.loads(json_str)
    list_train.append(result)

datos = {}

for l in list_train:
    datos[l['ctx']] = []
    for i in range(4):
        datos[l['ctx']].append(l['endings'][i])
    
    datos[l['ctx']].append(l['label'])


###### 


breakpoint()
encoded = []
for d in datos.keys():
    prompt = d
    choice0 = datos[d][0]
    choice1 = datos[d][1]
    choice2 = datos[d][2]
    choice3 = datos[d][3]

    label = torch.tensor(datos[d][4]).unsqueeze(0)

    encoding = tokenizer([prompt, prompt, prompt, prompt], [choice0, choice1, choice2, choice3], return_tensors='pt', padding=True)
    encoded.append(encoding)
    outputs = model(**{k: v.unsqueeze(0) for k, v in encoding.items()}, labels=label) 


    loss = outputs.loss
    logits = outputs.logits

    predictions = nn.functional.softmax(logits, dim=-1) 

######## TRAINING ########

training_args = TrainingArguments(
    output_dir="path/to/save/folder/",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
)

tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
model = RobertaForMultipleChoice.from_pretrained("roberta-large")

#tecnicamente aca van los datos.
#train_dataset = dataset["train"] 
#eval_dataset = dataset["eval"]
"""

training_args = TrainingArguments(
    output_dir="./",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
)

tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
model = RobertaForMultipleChoice.from_pretrained("roberta-large")

dataset_train = load_dataset("json", 'regular', data_files="hellaswag/data/hellaswag_train.jsonl")
dataset_val = load_dataset("json", 'regular', data_files="hellaswag/data/hellaswag_val.jsonl")

def preprocess_function(examples):
    breakpoint()
    #first_sentences = [[context] * 4 for context in examples["ctx"]]
    #second_sentences = [examples['endings'][i] for i in range(4)]
    
    first_sentences = [examples["ctx"], examples["ctx"], examples["ctx"], examples["ctx"]]
    second_sentences = [examples['endings'][i] for i in range(4)]

    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
    return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

breakpoint()

tokenized_swag = dataset_train['train'].map(preprocess_function, batched=True)

class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch

data_collator = DataCollatorForMultipleChoice() 


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train,
    eval_dataset=valid,
    tokenizer=tokenizer,
    data_collator=data_collator,
) 

trainer.train()
