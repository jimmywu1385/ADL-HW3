# Homework 3 - Summarization

## Environment
same as environment provided by TA and environment at https://github.com/moooooser999/ADL22-HW3
or you can install package by
```
pip install -r requirements.txt
```
## setting
download all model
```
bash download.sh
```

## Training
```
python3.9 train.py --max_input 256 --max_output 64 --batch_size 4 --accum_size 4 --ckpt_dir ./ckpt
```

## Testing 
```
bash ./run.sh ./path/to/test.jsonl ./path/to/output.jsonl
```