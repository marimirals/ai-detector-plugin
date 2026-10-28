"""Межъязыковой перенос детекторов на признаках (серия Э2): стилометрический, статистический и гибридный.
Английский и русский детекторы используют разные языковые модели (GPT-2 и ruGPT-3), но одинаковый набор признаков,
поэтому классификатор, обученный на английских признаках, можно применить к русским."""
import numpy as np
from hybrid import fit, VARIANTS
from metrics_utils import evaluate


def moments(F):
    """Среднее и стандартное отклонение каждого признака (метки не нужны)."""
    mu = F.mean(0); sd = F.std(0); sd[sd == 0] = 1.0
    return mu, sd


def align(F_target, src_moments, tgt_moments):
    """Нормировка по языку: распределение каждого признака целевого языка приводится к среднему и разбросу
    исходного языка. Используются только неразмеченные тексты целевого языка."""
    (mu_s, sd_s), (mu_t, sd_t) = src_moments, tgt_moments
    return (F_target - mu_t) / sd_t * sd_s + mu_s


def zero_shot(F_en_tr, y_en_tr, F_ru_tr, F_ru_te, y_ru_te, variants=VARIANTS, n_boot=200):
    """Обучение только на английских текстах, проверка на русских: без адаптации и с нормировкой по языку."""
    out, models = {}, {}
    for name, cols in variants.items():
        m = fit(F_en_tr[cols].values, y_en_tr); models[name] = m
        Xte = F_ru_te[cols].values
        out[name] = {'без адаптации': evaluate(y_ru_te, m.predict_proba(Xte)[:, 1], n_boot=n_boot)}
        Xal = align(Xte, moments(F_en_tr[cols].values), moments(F_ru_tr[cols].values))
        out[name]['нормировка по языку'] = evaluate(y_ru_te, m.predict_proba(Xal)[:, 1], n_boot=n_boot)
    return out, models


def few_shot(F_en_tr, y_en_tr, F_ru_tr, y_ru_tr, F_ru_te, y_ru_te, idx, variants=VARIANTS):
    """idx — индексы n русских обучающих примеров. Варианты:
    'только русские' — классификатор обучается на n русских примерах;
    'англ. + русские' — на всех английских и n русских примерах, русские признаки нормированы по языку
    (по всем неразмеченным русским обучающим текстам), суммарный вес русских примеров равен весу английских."""
    out = {}
    y_ru_tr = np.asarray(y_ru_tr); y_en_tr = np.asarray(y_en_tr)
    for name, cols in variants.items():
        Xr = F_ru_tr[cols].values; Xe = F_en_tr[cols].values; Xte = F_ru_te[cols].values
        r = {}
        m = fit(Xr[idx], y_ru_tr[idx])
        r['только русские'] = evaluate(y_ru_te, m.predict_proba(Xte)[:, 1], n_boot=0)
        mom_en, mom_ru = moments(Xe), moments(Xr)
        X = np.vstack([Xe, align(Xr[idx], mom_en, mom_ru)])
        y = np.concatenate([y_en_tr, y_ru_tr[idx]])
        w = np.concatenate([np.ones(len(Xe)), np.full(len(idx), len(Xe) / len(idx))])
        m2 = fit(X, y, sample_weight=w)
        r['англ. + русские'] = evaluate(y_ru_te, m2.predict_proba(align(Xte, mom_en, mom_ru))[:, 1], n_boot=0)
        out[name] = r
    return out
