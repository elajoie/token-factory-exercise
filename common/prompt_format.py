"""Single source of truth for the prompt format.

The rendering below mirrors bfcl_eval's QwenHandler._format_prompt (prompting mode)
byte-for-byte, so training, BFCL evaluation, benchmarking and production serving all
see exactly the same token sequence. A train/eval format mismatch is the #1 silent
accuracy killer in function-calling fine-tunes; tests/test_prompt_parity.py guards it.

Output format taught to the model (ToolACE == BFCL prompting mode):
    [func_name1(param=value, ...), func_name2(...)]
"""

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

# NOTE: the BFCL system prompt itself is NOT hard-coded here. BFCL assembles it from
# templates (bfcl_eval.model_handler.utils.formulate_system_prompt), so we generate the
# production chat template from BFCL's own code (serve/build_chat_template.py) to keep
# the exact same wording, whitespace and JSON indentation.


def render_segments(system, turns):
    """Render a conversation into (text, trainable) segments.

    system: str or None
    turns:  list of {"role": "user"|"assistant"|"tool", "content": str}

    Only assistant content + its <|im_end|> is trainable (loss masking).
    """
    segs = []
    if system is not None:
        segs.append((f"{IM_START}system\n{system}{IM_END}\n", False))
    n = len(turns)
    for idx, t in enumerate(turns):
        role, content = t["role"], t["content"]
        if role == "user":
            segs.append((f"{IM_START}user\n{content}{IM_END}\n", False))
        elif role == "assistant":
            segs.append((f"{IM_START}assistant\n", False))
            segs.append((f"{content}{IM_END}", True))
            segs.append(("\n", False))
        elif role == "tool":
            prev_role = turns[idx - 1]["role"] if idx > 0 else None
            next_role = turns[idx + 1]["role"] if idx < n - 1 else None
            s = ""
            if prev_role != "tool":
                s += f"{IM_START}user"
            s += f"\n<tool_response>\n{content}\n</tool_response>"
            if next_role != "tool":
                s += f"{IM_END}\n"
            segs.append((s, False))
        else:
            raise ValueError(f"unknown role: {role}")
    return segs


def render_prompt(system, turns):
    """Full inference prompt: history + generation prompt (what BFCL sends)."""
    return "".join(s for s, _ in render_segments(system, turns)) + f"{IM_START}assistant\n"
