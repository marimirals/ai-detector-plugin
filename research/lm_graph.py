"""Вычисление величин по токенам внутри графа модели ONNX (шаг 6).
Раньше модель возвращала логиты — по 50 тысяч чисел на каждый токен, и признаки считались по ним на JavaScript.
Теперь граф сам вычисляет для каждого токена 4 числа: log p, энтропию H, дисперсию log p и ранг."""
import os, time, numpy as np


def make_feature_model(lm):
    import torch

    class FeatureHead(torch.nn.Module):
        def __init__(self, lm):
            super().__init__(); self.lm = lm

        def forward(self, input_ids, attention_mask):
            logits = self.lm(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[:, :-1, :].float()
            target = input_ids[:, 1:].unsqueeze(-1)
            logp = torch.log_softmax(logits, dim=-1)
            lp = torch.gather(logp, -1, target).squeeze(-1)                      # log P(w_i | w_<i)
            p = torch.exp(logp)
            H = -(p * logp).sum(-1)                                              # энтропия
            var = (p * (logp + H.unsqueeze(-1)) ** 2).sum(-1)                    # дисперсия log p (устойчивая форма)
            rank = (logp > lp.unsqueeze(-1)).sum(-1).to(torch.float32) + 1.0     # ранг фактического токена
            return torch.stack([lp, H, var, rank], dim=-1)                      # [batch, T-1, 4]
    return FeatureHead(lm)


def export_feature_graph(lm, tok, out_dir, opset=17):
    """Экспорт графа «модель + величины по токенам» (вход: input_ids, attention_mask; выход: stats) и токенизатора."""
    import torch
    os.makedirs(os.path.join(out_dir, 'onnx'), exist_ok=True)
    lm = lm.float().cpu().eval(); lm.config.use_cache = False
    m = make_feature_model(lm).eval()
    ids = torch.ones(1, 8, dtype=torch.long); mask = torch.ones(1, 8, dtype=torch.long)
    path = os.path.join(out_dir, 'onnx', 'model.onnx')
    kw = dict(input_names=['input_ids', 'attention_mask'], output_names=['stats'], opset_version=opset,
              dynamic_axes={'input_ids': {0: 'batch', 1: 'seq'}, 'attention_mask': {0: 'batch', 1: 'seq'},
                            'stats': {0: 'batch', 1: 'seq_minus_1'}})
    with torch.no_grad():
        try:
            torch.onnx.export(m, (ids, mask), path, dynamo=False, **kw)
        except TypeError:
            torch.onnx.export(m, (ids, mask), path, **kw)
    lm.config.save_pretrained(out_dir); tok.save_pretrained(out_dir)
    return path


def graph_token_stats(session, tok, texts, max_tokens=256):
    """Величины по токенам из графа (по одному тексту, как в браузере). Время — полное: модель + разбор ответа."""
    out, times = [], []
    for t in texts:
        ids = tok(t, add_special_tokens=False, truncation=True, max_length=max_tokens, return_tensors='np')['input_ids'].astype(np.int64)
        if ids.shape[1] < 2:
            out.append({'ids': ids[0], 'lp': np.array([]), 'ent': np.array([]), 'var': np.array([]), 'rank': np.array([])}); continue
        t0 = time.perf_counter()
        st = session.run(['stats'], {'input_ids': ids, 'attention_mask': np.ones_like(ids)})[0][0].astype(np.float64)
        rec = {'ids': ids[0], 'lp': st[:, 0], 'ent': st[:, 1], 'var': st[:, 2], 'rank': st[:, 3].round().astype(np.int64)}
        times.append(time.perf_counter() - t0)
        out.append(rec)
    return out, times


def logits_token_stats_timed(session, tok, texts, max_tokens=256):
    """Прежний способ (граф отдаёт логиты, величины считаются снаружи) — с полным временем на текст."""
    from onnx_utils import ort_token_stats
    out, times = [], []
    for t in texts:
        t0 = time.perf_counter()
        st, _ = ort_token_stats(session, tok, [t], max_tokens)
        times.append(time.perf_counter() - t0); out.append(st[0])
    return out, times
