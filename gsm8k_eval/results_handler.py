# gsm8k_eval/results_handler.py
import os
import json

class ResultsFileHandler:
    """
    Manages the results JSONL file for GSM8K evaluation:
      - Loads existing results for resumption
      - Tracks how many examples are correct so far
      - Appends new results
      - Closes the file
    """
    def __init__(self, results_file):
        self.results_file = results_file
        self.existing_data = {}
        self.max_id = -1
        self.correct_so_far = 0

        # Load any existing results for resumption
        self._load_existing_data()

        # Open the file in append mode
        self.file = open(self.results_file, "a", encoding="utf-8")

    def _load_existing_data(self):
        """If the JSONL file exists, read it line by line and update internal state."""
        if os.path.exists(self.results_file):
            with open(self.results_file, "r", encoding="utf-8") as f:
                for line in f:
                    record_str = line.strip()
                    if not record_str:
                        continue
                    record = json.loads(record_str)
                    row_id = record["id"]
                    self.existing_data[row_id] = record

                    # Track highest processed ID
                    if row_id > self.max_id:
                        self.max_id = row_id

                    # Track how many were correct
                    if record.get("correct"):
                        self.correct_so_far += 1

    def get_start_index(self):
        """Returns the row index to start from when resuming."""
        return self.max_id + 1

    def get_correct_so_far(self):
        """Returns the number of correct answers so far (from old + new)."""
        return self.correct_so_far

    def append_result(self, record):
        """
        Appends a single record to the JSONL file, then flushes.
        Also updates correct_so_far if record is correct.
        """
        self.file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.file.flush()

        # Update correct_so_far if needed
        if record.get("correct"):
            self.correct_so_far += 1

    def close(self):
        """Closes the JSONL file. Should be called after evaluation is complete."""
        self.file.close()
