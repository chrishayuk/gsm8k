# gsm8k_eval/answer_parser.py
import re

def parse_answer(text):
    """
    Extract the last numeric value from a string.
    GSM8K answers often contain a line like "#### 42" or "42" at the end.
    
    Returns:
        str or None: The last numeric token found in `text`, or None if none found.
    """
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    if not matches:
        return None
    return matches[-1]
