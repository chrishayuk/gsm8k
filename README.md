uv run python -m gsm8k_eval.main \
  --model_name_or_path ibm-granite/granite-3.1-2b-instruct \
  --split test \
  --cot \
  --do_sample \
  --temperature 0.7 \
  --top_p 0.9 \
  --max_new_tokens 256
