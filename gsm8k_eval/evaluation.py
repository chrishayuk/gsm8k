# gsm8k_eval/evaluation.py
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from gsm8k_eval.results_handler import ResultsFileHandler
from gsm8k_eval.answer_parser import parse_answer
from gsm8k_eval.generation import generate_model_answer
from gsm8k_eval.display_utils import print_interim_debug, print_final_results

def evaluate_example(row_id, example, model, tokenizer, args, correct_so_far):
    """
    Evaluate a single example from the GSM8K dataset.

    Returns a dict 'record' and the updated correct_so_far:
      record = {
        "id": row_id,
        "question": str,
        "gold_answer": str,
        "parsed_gold_answer": str or None,
        "predicted_answer_text": str,
        "parsed_predicted_answer": str or None,
        "correct": bool,
        "running_accuracy": float
      }
      updated_correct_so_far = int
    """
    question = example["question"]
    gold_answer_full = example["answer"]  # chain-of-thought + final numeric

    # Parse the final numeric from the gold answer
    parsed_gold_answer = parse_answer(gold_answer_full)

    # Optionally use chain-of-thought prompting
    if args.cot:
        prompt = f"{question}\nLet's solve this step by step:\n"
    else:
        prompt = question

    # Generate text from the model
    generated_text = generate_model_answer(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p
    )

    # Parse the final numeric from the model's generated text
    parsed_predicted_answer = parse_answer(generated_text)

    # Check correctness
    is_correct = False
    if parsed_gold_answer and parsed_predicted_answer:
        is_correct = (parsed_predicted_answer == parsed_gold_answer)

    # The new correct_so_far is old + 1 if correct
    updated_correct_so_far = correct_so_far + (1 if is_correct else 0)
    processed_count = row_id + 1
    running_accuracy = updated_correct_so_far / processed_count * 100

    # Store everything in the record
    record = {
        "id": row_id,
        "question": question,
        "gold_answer": gold_answer_full,            # full chain-of-thought text
        "parsed_gold_answer": parsed_gold_answer,   # numeric from gold
        "predicted_answer_text": generated_text,    # full model output text
        "parsed_predicted_answer": parsed_predicted_answer,  # numeric from model output
        "correct": is_correct,
        "running_accuracy": running_accuracy
    }
    return record, updated_correct_so_far

def run_evaluation(args):
    """
    Main evaluation loop:
      1. Load dataset/model
      2. Create ResultsFileHandler for resumption
      3. Evaluate each unprocessed example
      4. Append results, print debug
      5. Close file, print final results
    """
    # Load dataset
    dataset = load_dataset("gsm8k", "main")
    data = dataset[args.split]
    total = len(data)

    # Create a results file handler (loads existing results)
    results_handler = ResultsFileHandler(args.results_file)
    start_idx = results_handler.get_start_index()
    correct_so_far = results_handler.get_correct_so_far()

    # Load model/tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.to(device)
    model.eval()

    for row_id in range(start_idx, total):
        example = data[row_id]

        # Evaluate this example
        record, correct_so_far = evaluate_example(
            row_id, example, model, tokenizer, args, correct_so_far
        )

        # Write the record out (JSONL)
        results_handler.append_result(record)

        # Print debug info at intervals or at the last
        processed_count = row_id + 1
        if (processed_count % args.print_frequency == 0) or (processed_count == total):
            print_interim_debug(record, processed_count, total)

    # Close the results file
    results_handler.close()

    # Print final results
    print_final_results(correct_so_far, total)

