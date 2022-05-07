from argparse import ArgumentParser, Namespace
from pathlib import Path

import nltk 
import numpy as np
from datasets import load_metric
from transformers import T5Tokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

from dataset import S2SData

def main(args):
    nltk.download('punkt')

    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_path)

    train_dataset = S2SData(args.train_path, tokenizer, args.max_input, args.max_output, "train", args.prefix)
    eval_dataset = S2SData(args.eval_path, tokenizer, args.max_input, args.max_output, "eval", args.prefix)

    metric = load_metric("rouge")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.ckpt_dir,
        overwrite_output_dir=True,
        save_strategy="no",
        evaluation_strategy="step",
        logging_strategy="step",
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
        compute_metrics=compute_metrics,
    )

    trainer.train()

    tokenizer.save_pretrained(args.ckpt_dir)
    model.save_pretrained(args.ckpt_dir)

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
    parser.add_argument(
        "--prefix",
        type=str,
        help="input prefix.",
        default="summarize: ",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    main(args)