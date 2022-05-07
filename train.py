from argparse import ArgumentParser, Namespace
from pathlib import Path

from transformers import T5Tokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np

from dataset import S2SData
#from tw_rouge import get_rouge

#import tensorflow as tf
#tf.config.set_visible_devices([], 'GPU')

def main(args):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_path)

    train_dataset = S2SData(args.train_path, tokenizer, args.max_input, args.max_output, "train")
    eval_dataset = S2SData(args.eval_path, tokenizer, args.max_input, args.max_output, "eval")
    '''
    def compute_metrics(eval_preds):
        def flatten_dict(d, prefixes=()):
            ret = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    ret |= flatten_dict(v, prefixes=prefixes + (k,))
                elif isinstance(v, (int, float)):
                    ret |= {"_".join(prefixes + (k,)): v}
                else:
                    raise ValueError
            return ret
        pred_text = []
        label_text = []
        for p, l in zip(eval_preds.predictions, eval_preds.label_ids):
                pred = tokenizer.decode(p, clean_up_tokenization_spaces=True)
                pred_stop = pred.find(tokenizer.eos_token)
                if pred_stop is not None:
                    pred = pred[:pred_stop]
                pred_text.append(pred)

                label = tokenizer.decode(l, clean_up_tokenization_spaces=True)
                label_stop = label.find(tokenizer.eos_token)
                if label_stop is not None:
                    label = label[:label_stop]
                label_text.append(label)   

        rouge = get_rouge(pred_text, label_text, avg=True, ignore_empty=False)
        return flatten_dict(rouge)
    '''
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.ckpt_dir,
        overwrite_output_dir=True,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_strategy="epoch",
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

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-5)

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accum_size", type=int, default=8)

    parser.add_argument(
        "--max_input", type=int, help="input len", default=256
    )
    parser.add_argument(
        "--max_output", type=int, help="output len", default=64
    )    

    parser.add_argument("--num_epoch", type=int, default=20)
    parser.add_argument("--use_adafactor", action="store_true")
    parser.add_argument("--no_fp16", dest="fp16", action="store_false")

    parser.add_argument(
        "--pretrained_path",
        type=str,
        help="model path.",
        default="google/mt5-small",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    main(args)