from typing import Dict
import json

from torch.utils.data import Dataset

class S2SData(Dataset):
    def __init__(
        self,
        data_path,
        prefix,
        tokenizer,
        max_input,
        max_output,
    ):
        self.data = []
        for line in data_path.read_text().split("\n")[:-1]:
            data = json.loads(line)
            self.data.append(data)
            
        self.prefix = prefix
        self.tokenizer = tokenizer
        self.max_input = max_input
        self.max_output = max_output

    def __len__(self)->int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        inputs = self.prefix + self.data[index]["maintext"]
        model_inputs = self.tokenizer(inputs, max_length=self.max_input, padding="max_length",
                                        truncation=True)

        title = self.data[index].get("title", None)
        if title is not None:
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(self.data[index]["title"], max_length=self.max_output, padding="max_length",
                                        truncation=True)

            model_inputs["labels"] = labels["input_ids"]

        return model_inputs