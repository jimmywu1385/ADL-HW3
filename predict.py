from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from transformers import T5Tokenizer
from transformers import AutoModelForSeq2SeqLM
from tqdm import tqdm
import json

from dataset import S2SData

def set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(args):
    set_random(args.random_seed)

    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_path)

    test_dataset = S2SData(args.test_path, tokenizer, args.max_input, args.max_output, "test")
    test_datasets = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model.to(args.device)
    model.eval()

    result = []
    with torch.no_grad():
        for i, dic in enumerate(test_datasets):
            outputs = model.generate(
                input_ids=dic["input_ids"].to(args.device),
                attention_mask=dic['attention_mask'].to(args.device),
                max_length=args.max_output,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            output_sequences = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for output_seq, ID in zip(output_sequences, dic["id"]):
                result.append({"title": output_seq, "id": ID})

    with open(args.output_path, "w") as f:
        for r in result:
            print(json.dumps(r, ensure_ascii=False), file=f)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=123)

    parser.add_argument(
        "--test_path",
        type=Path,
        help="Directory to the data.",
        default="data/sample_test.jsonl",
    )

    parser.add_argument(
        "--output_path",
        type=Path,
        help="Directory to the data.",
        default="./qq.jsonl",
    )
    parser.add_argument("--batch_size", type=int, default=4)

    parser.add_argument(
        "--max_input", type=int, help="input len", default=256
    )
    parser.add_argument(
        "--max_output", type=int, help="output len", default=64
    )    

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        help="model path.",
        default="./ckpt/checkpoint-1",
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)