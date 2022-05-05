
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from transformers import AutoTokenizer, T5Tokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
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

    train_dataset = S2SData(args.train_path, args.prefix, tokenizer, args.max_input, args.max_output)
    eval_dataset = S2SData(args.eval_path, args.prefix, tokenizer, args.max_input, args.max_output)
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.ckpt_dir,
        overwrite_output_dir=True,
        save_strategy='epoch',
        seed=args.random_seed,
        fp16=args.fp16,
        num_train_epochs=args.num_epoch,
        adafactor=args.use_adafactor,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accum_size,
        predict_with_generate=True,
        logging_dir=args.log_dir,
    )


    trainer = Seq2SeqTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=123)

    parser.add_argument(
        "--train_path",
        type=Path,
        help="Directory to the data.",
        default="data/train.jsonl",
    )
    parser.add_argument(
        "--eval_path",
        type=Path,
        help="Directory to the data.",
        default="data/public.jsonl",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt",
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        help="Directory to save the log file.",
        default="./log",
    )

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accum_size", type=int, default=2)

    parser.add_argument(
        "--max_input", type=int, help="input len", default=256
    )
    parser.add_argument(
        "--max_output", type=int, help="output len", default=64
    )    

    parser.add_argument("--num_epoch", type=int, default=2)
    parser.add_argument("--use_adafactor", action="store_true")
    parser.add_argument("--no_fp16", dest="fp16", action="store_false")

    parser.add_argument(
        "--pretrained_path",
        type=str,
        help="model path.",
        default="google/mt5-small",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        help="T5 prefix.",
        default="summarize: ",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    main(args)