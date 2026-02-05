import re


def build_formatted_prompt(tokenizer, user_prompt: str, add_generation_prompt: bool = True) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
    )


def extract_user_instruction(s: str) -> str:
    """
    Extract the user instruction between:
      <|start_header_id|>user<|end_header_id|>  ...  <|eot_id|>
    Returns "" if not found.
    """
    pattern = r"<\|start_header_id\|>user<\|end_header_id\|>\s*(.*?)\s*<\|eot_id\|>"
    m = re.search(pattern, s, flags=re.DOTALL)
    return m.group(1).strip() if m else ""
