"""Экспорт языковых моделей в ONNX, квантование INT8 и вычисление признаков через ONNX Runtime (разделы 1.5, 4.1)."""
import os, time, numpy as np

def export_onnx(model, tok, out_dir, opset=17):
    """Экспорт модели (вход: input_ids, attention_mask; выход: logits) и токенизатора в формат для Transformers.js."""
    import torch
    os.makedirs(os.path.join(out_dir, 'onnx'), exist_ok=True)
    model = model.float().cpu().eval()
    model.config.use_cache = False

    class Wrap(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits

    ids = torch.ones(1, 8, dtype=torch.long); mask = torch.ones(1, 8, dtype=torch.long)
    path = os.path.join(out_dir, 'onnx', 'model.onnx')
    kw = dict(input_names=['input_ids', 'attention_mask'], output_names=['logits'], opset_version=opset,
              dynamic_axes={'input_ids': {0: 'batch', 1: 'seq'}, 'attention_mask': {0: 'batch', 1: 'seq'},
                            'logits': {0: 'batch', 1: 'seq'}})
    try:
        torch.onnx.export(Wrap(model), (ids, mask), path, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(Wrap(model), (ids, mask), path, **kw)
    model.config.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)          # быстрый токенизатор сохраняет tokenizer.json
    return path

def quantize(path_fp32):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    out = path_fp32.replace('model.onnx', 'model_quantized.onnx')
    quantize_dynamic(path_fp32, out, weight_type=QuantType.QInt8)
    return out

def size_mb(path):
    return os.path.getsize(path) / 2**20

def ort_token_stats(session, tok, texts, max_tokens=256):
    """То же, что lm_features.token_stats, но через ONNX Runtime на CPU — по одному тексту, как в браузере."""
    out, times = [], []
    for t in texts:
        ids = tok(t, add_special_tokens=False, truncation=True, max_length=max_tokens, return_tensors='np')['input_ids'].astype(np.int64)
        if ids.shape[1] < 2:
            out.append({'ids': ids[0], 'lp': np.array([]), 'ent': np.array([]), 'var': np.array([]), 'rank': np.array([])}); continue
        t0 = time.perf_counter()
        logits = session.run(['logits'], {'input_ids': ids, 'attention_mask': np.ones_like(ids)})[0][0, :-1].astype(np.float64)
        times.append(time.perf_counter() - t0)
        mx = logits.max(1, keepdims=True)
        logp = logits - (np.log(np.exp(logits - mx).sum(1, keepdims=True)) + mx)
        tgt = ids[0, 1:]
        lp = logp[np.arange(len(tgt)), tgt]
        p = np.exp(logp)
        ent = -(p * logp).sum(1)
        var = (p * logp ** 2).sum(1) - ent ** 2
        rank = (logp > lp[:, None]).sum(1) + 1
        out.append({'ids': ids[0], 'lp': lp, 'ent': ent, 'var': var, 'rank': rank})
    return out, times
