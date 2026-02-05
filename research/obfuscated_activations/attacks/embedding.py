import torch
import torch.nn.functional as F

from utils.probes import score_activations
from utils.data import build_formatted_prompt

OPT_LOC_TOKEN = "<|optim-location|>"


def ensure_opt_loc_token(tokenizer) -> int:
    if OPT_LOC_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [OPT_LOC_TOKEN]})
    return tokenizer.convert_tokens_to_ids(OPT_LOC_TOKEN)


def insert_suffix_embeddings(
    embedding_layer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    suffix_emb: torch.Tensor,
    opt_loc_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if suffix_emb.dim() == 2:
        suffix_emb = suffix_emb.unsqueeze(0)

    locations = (input_ids == opt_loc_token_id).nonzero(as_tuple=False)
    if locations.numel() == 0:
        raise ValueError("opt_loc_token_id not found in input_ids")

    counts = torch.sum(input_ids == opt_loc_token_id, dim=1)
    if not torch.all(counts == 1):
        raise ValueError("Expected exactly one opt_loc_token_id per row")

    unique_columns = torch.unique(locations[:, 1])
    if len(unique_columns) != 1:
        raise ValueError("opt_loc_token_id is not in the same column in all rows")

    insertion_column = unique_columns.item()
    left_input_ids = input_ids[:, :insertion_column]
    right_input_ids = input_ids[:, insertion_column + 1 :]
    left_attention_mask = attention_mask[:, :insertion_column]
    right_attention_mask = attention_mask[:, insertion_column + 1 :]

    left_input_embeds = embedding_layer(left_input_ids)
    right_input_embeds = embedding_layer(right_input_ids)

    suffix_emb = suffix_emb.to(
        device=embedding_layer.weight.device,
        dtype=embedding_layer.weight.dtype,
    )
    suffix_batch = suffix_emb.expand(input_ids.shape[0], -1, -1)

    input_embeds = torch.cat([left_input_embeds, suffix_batch, right_input_embeds], dim=1)
    suffix_mask = torch.ones(
        (attention_mask.shape[0], suffix_batch.shape[1]),
        device=attention_mask.device,
        dtype=attention_mask.dtype,
    )
    attention_mask = torch.cat([left_attention_mask, suffix_mask, right_attention_mask], dim=1)

    return input_embeds, attention_mask


def init_suffix_embeddings(embedding_weight: torch.Tensor, suffix_len: int, seed: int) -> torch.nn.Parameter:
    gen = torch.Generator(device=embedding_weight.device)
    gen.manual_seed(seed)
    vocab_size = embedding_weight.shape[0]
    token_ids = torch.randint(0, vocab_size, (suffix_len,), generator=gen, device=embedding_weight.device)
    init = embedding_weight[token_ids].detach().clone()
    return torch.nn.Parameter(init.unsqueeze(0))


def compute_example_loss(
    tokenizer,
    model,
    embedding_layer,
    probe_params: dict[int, dict],
    prompt_text: str,
    completion_text: str,
    suffix_emb: torch.Tensor,
    probe_type: str,
    lambda_behavior: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not 0.0 <= lambda_behavior <= 1.0:
        raise ValueError("lambda_behavior must be in [0, 1]")

    opt_loc_token_id = ensure_opt_loc_token(tokenizer)
    formatted_with_gen = build_formatted_prompt(
        tokenizer,
        prompt_text + OPT_LOC_TOKEN,
        add_generation_prompt=True,
    )

    device = embedding_layer.weight.device
    prompt_ids = tokenizer(formatted_with_gen, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    prompt_mask = torch.ones_like(prompt_ids, dtype=torch.long)
    prompt_embeds, _ = insert_suffix_embeddings(
        embedding_layer=embedding_layer,
        input_ids=prompt_ids,
        attention_mask=prompt_mask,
        suffix_emb=suffix_emb,
        opt_loc_token_id=opt_loc_token_id,
    )

    completion_ids = tokenizer(
        completion_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)
    completion_embeds = embedding_layer(completion_ids)

    inputs_embeds = torch.cat([prompt_embeds, completion_embeds], dim=1)
    attention_mask = torch.ones(inputs_embeds.shape[:2], device=device, dtype=torch.long)
    labels = torch.full((1, inputs_embeds.shape[1]), -100, device=device, dtype=torch.long)
    labels[:, prompt_embeds.shape[1]:] = completion_ids

    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
        output_hidden_states=True,
        use_cache=False,
    )

    hidden_states = outputs.hidden_states
    behavior_loss = outputs.loss
    activations_by_layer = {
        layer_idx: hidden_states[layer_idx + 1][0] for layer_idx in range(len(probe_params))
    }
    gen_start = prompt_embeds.shape[1]
    completion_len = completion_ids.shape[1]
    score = score_activations(
        activations_by_layer,
        probe_params,
        probe_type,
        gen_start=gen_start,
        completion_len=completion_len,
    )

    obf_loss = score
    total_loss = lambda_behavior * behavior_loss + (1.0 - lambda_behavior) * obf_loss
    return total_loss, behavior_loss, score
