# gsm8k_eval/args.py
import argparse

def parse_arguments():
    """Parse command-line arguments and return the args object."""
    parser = argparse.ArgumentParser(description="Evaluate a model on GSM8K with optional chain-of-thought prompting.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Local path or HF Hub model ID (e.g., 'gpt2' or '/path/to/model')."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Which split of GSM8K to evaluate on (train or test)."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Max tokens to generate for each question."
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Use sampling for generation (else greedy)."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (ignored if do_sample=False)."
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling (ignored if do_sample=False)."
    )
    parser.add_argument(
        "--cot",
        action="store_true",
        help="Use chain-of-thought prompting: 'Let's solve this step by step:'."
    )
    parser.add_argument(
        "--print_frequency",
        type=int,
        default=5,
        help="Print debug info after every N examples."
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default="results.jsonl",
        help="JSONL file where we store results (allows resumption)."
    )
    return parser.parse_args()
