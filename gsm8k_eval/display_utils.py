# gsm8k_eval/display_utils.py

def print_interim_debug(record, processed_count, total):
    """
    Prints an interim debug block for a single example (record).
    Includes question, gold answer (full text), parsed gold answer,
    the full predicted text, parsed predicted answer, correctness,
    and running accuracy.
    """
    print(f"\n--- Example {processed_count}/{total} ---")
    print("Question:")
    print(record["question"])

    print("\nGold Answer (full):")
    print(record["gold_answer"])

    print("\nParsed Gold Numeric:")
    print(record["parsed_gold_answer"])

    print("\nPredicted Answer Text:")
    print(record["predicted_answer_text"])

    print("\nParsed Predicted Numeric:")
    print(record["parsed_predicted_answer"])

    print(f"\nCorrect? {record['correct']}")
    print(f"Running Accuracy: {record['running_accuracy']:.2f}%")
    print("-" * 60)

def print_final_results(correct_so_far, total):
    """
    Prints a final summary after all examples are processed.
    """
    final_accuracy = correct_so_far / total * 100 if total > 0 else 0.0
    print("\n=== Final Results ===")
    print(f"Processed examples: {total}")
    print(f"Correct predictions: {correct_so_far}")
    print(f"Final Accuracy: {final_accuracy:.2f}%")

