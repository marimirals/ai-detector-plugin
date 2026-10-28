"""Дообучение многоязычного энкодера (XLM-RoBERTa) для обнаружения сгенерированного текста (серия Э2).
Обычный цикл обучения PyTorch: так меньше зависимости от версий библиотек, чем с Trainer."""
import math, time, random
import numpy as np


def set_seed(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def encode(tok, texts, max_len=256):
    """Токенизация один раз заранее: список списков идентификаторов токенов (без дополнения)."""
    return tok(list(texts), truncation=True, max_length=max_len, add_special_tokens=True)['input_ids']


def _batch(ids_list, pad_id, device):
    import torch
    L = max(len(x) for x in ids_list)
    ids = torch.full((len(ids_list), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(ids_list), L), dtype=torch.long)
    for i, x in enumerate(ids_list):
        ids[i, :len(x)] = torch.tensor(x, dtype=torch.long); mask[i, :len(x)] = 1
    return ids.to(device), mask.to(device)


def train(init, ids, labels, pad_id, steps=None, epochs=None, batch_size=16, lr=2e-5, warmup=0.1,
          weight_decay=0.01, seed=0, device='cuda', log_every=100):
    """Дообучает классификатор. init — имя модели на Hugging Face или путь к сохранённой модели.
    Длительность задаётся либо числом шагов (steps), либо числом эпох (epochs)."""
    import torch
    from transformers import AutoModelForSequenceClassification
    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(init, num_labels=2).to(device)
    labels = np.asarray(labels, dtype=np.int64); n = len(labels)
    per_epoch = math.ceil(n / batch_size)
    total = steps if steps is not None else per_epoch * epochs
    no_decay = ('bias', 'LayerNorm.weight', 'layer_norm.weight')
    groups = [{'params': [p for nm, p in model.named_parameters() if not any(k in nm for k in no_decay)], 'weight_decay': weight_decay},
              {'params': [p for nm, p in model.named_parameters() if any(k in nm for k in no_decay)], 'weight_decay': 0.0}]
    opt = torch.optim.AdamW(groups, lr=lr)
    n_warm = max(1, int(total * warmup))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / n_warm if s < n_warm else max(0.0, (total - s) / max(1, total - n_warm)))
    use_amp = device == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    rng = np.random.default_rng(seed)
    model.train(); step = 0; t0 = time.time(); run_loss = []
    while step < total:
        order = rng.permutation(n)                      # новая перестановка на каждую эпоху
        for b in range(0, n, batch_size):
            if step >= total: break
            idx = order[b:b + batch_size]
            x, m = _batch([ids[i] for i in idx], pad_id, device)
            y = torch.tensor(labels[idx], device=device)
            with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                logits = model(input_ids=x, attention_mask=m).logits
            loss = torch.nn.functional.cross_entropy(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            run_loss.append(loss.item()); step += 1
            if log_every and step % log_every == 0:
                print(f'    шаг {step}/{total}  loss={np.mean(run_loss[-log_every:]):.3f}  {time.time() - t0:.0f} с')
    model.eval()
    return model


def predict(model, ids, pad_id, batch_size=64, device='cuda'):
    """Вероятность класса «сгенерирован» (метка 1) для каждого текста."""
    import torch
    out = []
    order = np.argsort([len(x) for x in ids])          # близкие по длине тексты в одном пакете — быстрее
    with torch.no_grad():
        for b in range(0, len(ids), batch_size):
            idx = order[b:b + batch_size]
            x, m = _batch([ids[i] for i in idx], pad_id, device)
            with torch.autocast('cuda', dtype=torch.float16, enabled=device == 'cuda'):
                logits = model(input_ids=x, attention_mask=m).logits.float()
            out.append((idx, torch.softmax(logits, -1)[:, 1].cpu().numpy()))
    p = np.zeros(len(ids), dtype=np.float64)
    for idx, pr in out: p[idx] = pr
    return p


def free():
    """Освобождает память видеокарты. Перед вызовом удалите модель: del model."""
    import torch, gc
    gc.collect(); torch.cuda.empty_cache()


def few_shot_sample(labels, n, seed):
    """Индексы n примеров (по n/2 каждого класса) из обучающей выборки."""
    rng = np.random.default_rng(seed); labels = np.asarray(labels)
    pos = rng.choice(np.where(labels == 1)[0], n // 2, replace=False)
    neg = rng.choice(np.where(labels == 0)[0], n - n // 2, replace=False)
    return rng.permutation(np.concatenate([pos, neg]))
