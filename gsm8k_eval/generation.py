# gsm8k_eval/generation.py
import torch

def generate_model_answer(prompt, model, tokenizer, max_new_tokens=128, do_sample=False, temperature=0.7, top_p=0.9):
    """
    Given a prompt, model, tokenizer, and generation hyperparameters,
    returns the decoded response text from the model.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text
