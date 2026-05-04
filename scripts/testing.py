import torch


def _uniform_x0(B, L, K):
    # Sample logistic-normal: Gaussian in log space, then CLR
    noise = torch.randn((B, L, K))
    log_p0 = noise - torch.logsumexp(
        noise, dim=-1, keepdim=True
    )  # log-softmax → log-probs
    return log_p0  # - log_p0.mean(dim=-1, keepdim=True)  # CLR center


p0 = _uniform_x0(64, 128, 27)

print(torch.sum(torch.exp(p0[0][0])))
