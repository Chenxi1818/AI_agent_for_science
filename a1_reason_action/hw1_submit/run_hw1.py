
# ----------------------------------
# part1
python part1.py -model openai/gpt-oss-20b -n 20 --dataset_path openai/gsm8k
python part1.py -model openai/gpt-oss-120b -n 20 --dataset_path openai/gsm8k
python part1.py -model google/gemma-3-27b-it -n 20 --dataset_path openai/gsm8k

# --------------------------------
# part2
# --------------------------------
python part2.py -model openai/gpt-oss-20b -n 20 -s 1 --dataset_path openai/gsm8k
python part2.py -model openai/gpt-oss-20b -n 20 -s 5 --dataset_path openai/gsm8k
python part2.py -model openai/gpt-oss-20b -n 20 -s 10 --dataset_path openai/gsm8k

# change the structure of COT. part2_copy
# Create a new list of examples where the reasoning is extremely brief.
# the origin is "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72
# now is "In April Natalia sold 48 clips.\nIn May Natalia sold 24.\n72 altogether in April and May.\n#### 72"
python part2_copy.py -model openai/gpt-oss-20b -n 20 -s 5 --dataset_path openai/gsm8k

# ----------------------------
# part3: tool
# ----------------------------
python part3.py -model openai/gpt-oss-20b -n 20 --dataset_path openai/gsm8k
