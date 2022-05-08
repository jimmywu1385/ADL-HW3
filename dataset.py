from typing import Dict
import json

import torch
from torch.utils.data import Dataset

class S2SData(Dataset):
    def __init__(
        self,
        data_path,
        tokenizer,
        max_input,
        max_output,
        split,
        prefix,
    ):
        self.data = []
        for line in data_path.read_text().split("\n")[:-1]:
            data = json.loads(line)
            self.data.append(data)
            
        self.tokenizer = tokenizer
        self.max_input = max_input
        self.max_output = max_output
        self.split = split
        self.prefix = prefix

    def __len__(self)->int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        inputs = self.prefix + self.data[index]["maintext"]
        model_inputs = self.tokenizer(inputs[:50], max_length=self.max_input, padding="max_length",
                                        truncation=True, return_tensors="pt")

        if self.split == "test":
            model_inputs["id"] = self.data[index]["id"]      

        title = self.data[index].get("title", None)
        if title is not None:
            label_encode = self.tokenizer(title, max_length=self.max_output, padding="max_length",
                                    truncation=True).input_ids
            labels = torch.tensor(label_encode)
            labels[labels == self.tokenizer.pad_token_id] = -100

            model_inputs["labels"] = labels.flatten()

        model_inputs["input_ids"] = model_inputs["input_ids"].flatten()
        model_inputs["attention_mask"] = model_inputs["attention_mask"].flatten()

        return model_inputs