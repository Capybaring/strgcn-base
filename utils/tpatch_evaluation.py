"""t-PatchGNN compute_error/evaluation semantics on ragged observation nodes."""
import torch


def variable_error_sums(prediction, truth, mask, var_idx, num_vars):
    selected = mask.bool()
    # Fixed-size scatter avoids CUDA boolean indexing's dynamic output sizing.
    ids = torch.where(selected, var_idx, 0).long().reshape(-1)
    error = torch.where(selected, prediction - truth, 0.).reshape(-1)
    weight = selected.to(error.dtype).reshape(-1)
    count = torch.zeros(num_vars, dtype=error.dtype, device=error.device).scatter_add(0, ids, weight)
    se = torch.zeros_like(count).scatter_add(0, ids, error.square())
    ae = torch.zeros_like(count).scatter_add(0, ids, error.abs())
    return se, ae, count


def variable_macro_loss(prediction, truth, mask, var_idx, num_vars):
    se, _, count = variable_error_sums(prediction, truth, mask, var_idx, num_vars)
    # Upstream compute_error(reduce='mean'): mean over available variables.
    return (se / (count + 1e-8)).sum() / torch.count_nonzero(count).clamp_min(1)


def summarize_variable_errors(se, ae, count):
    available = torch.count_nonzero(count).clamp_min(1)
    mse = (se / (count + 1e-8)).sum() / available
    mae = (ae / (count + 1e-8)).sum() / available
    return dict(mse=mse.item(), mae=mae.item(), rmse=mse.sqrt().item())
