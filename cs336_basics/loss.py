import torch
from einops import reduce


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits_max = torch.amax(logits, dim=-1, keepdim=True)
    shifted_logits = logits - logits_max

    expsum = reduce(torch.exp(shifted_logits), '... vocab -> ...', 'sum')
    log_expsum = torch.log(expsum)

    target_logits = torch.gather(
        shifted_logits,
        dim=-1,
        index=targets.unsqueeze(-1)
    ).squeeze(-1)

    loss = log_expsum - target_logits
    return loss.mean()
