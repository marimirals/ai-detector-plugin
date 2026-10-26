"""Вероятностные признаки по компактной авторегрессионной языковой модели (раздел 2.3, формулы 2.1–2.2).
Для каждого токена считаются: логарифм вероятности, энтропия предсказанного распределения, ранг
фактического токена и дисперсия логарифма вероятности (для признака условной кривизны, как в Fast-DetectGPT)."""
import numpy as np

LM_FEATURE_NAMES = ['mean_lp', 'std_lp', 'mean_entropy', 'mean_log_rank',
                    'rank_1', 'rank_2_10', 'rank_11_100', 'rank_101_1000', 'rank_gt_1000', 'curvature']

def token_stats(texts, model, tok, max_tokens=256, batch_size=8, device='cuda'):
    """Один проход модели по каждому тексту. Возвращает список словарей с массивами по токенам."""
    import torch
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'right'
    out = []
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(batch, return_tensors='pt', truncation=True, max_length=max_tokens,
                  padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            logits = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask).logits.float()
        logp = torch.log_softmax(logits[:, :-1], dim=-1)          # прогноз для токенов 2..T
        target = enc.input_ids[:, 1:]
        lp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)    # log P(w_i | w_<i)
        p = logp.exp()
        ent = -(p * logp).sum(-1)                                  # энтропия, формула (2.2)
        var = (p * logp.pow(2)).sum(-1) - ent.pow(2)               # дисперсия log p по распределению модели
        rank = (logp > lp.unsqueeze(-1)).sum(-1) + 1               # ранг фактического токена
        mask = enc.attention_mask[:, 1:].bool()
        for b in range(len(batch)):
            m = mask[b]
            out.append({'ids': enc.input_ids[b][enc.attention_mask[b].bool()].cpu().numpy(),
                        'lp': lp[b][m].cpu().numpy(), 'ent': ent[b][m].cpu().numpy(),
                        'var': var[b][m].cpu().numpy(), 'rank': rank[b][m].cpu().numpy()})
        del logits, logp, p
    return out

def aggregate(s, k=None):
    """Признаки по первым k предсказанным токенам (k=None — все)."""
    lp, ent, var, rank = (s[x][:k] if k else s[x] for x in ('lp', 'ent', 'var', 'rank'))
    n = len(lp)
    if n == 0:
        return {f: 0.0 for f in LM_FEATURE_NAMES}
    curv = (lp.sum() + ent.sum()) / np.sqrt(max(var.sum(), 1e-8))   # (∑log p − ∑E[log p]) / √∑Var, E[log p] = −H
    return {
        'mean_lp': float(lp.mean()), 'std_lp': float(lp.std()), 'mean_entropy': float(ent.mean()),
        'mean_log_rank': float(np.log(rank).mean()),
        'rank_1': float((rank == 1).mean()), 'rank_2_10': float(((rank >= 2) & (rank <= 10)).mean()),
        'rank_11_100': float(((rank >= 11) & (rank <= 100)).mean()),
        'rank_101_1000': float(((rank >= 101) & (rank <= 1000)).mean()), 'rank_gt_1000': float((rank > 1000).mean()),
        'curvature': float(curv),
    }
