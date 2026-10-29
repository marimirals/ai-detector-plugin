"""Гибридный детектор, версия 2 (шаг 6): дополнительные вероятностные признаки, подбор регуляризации
на валидационной выборке, обучение на нескольких корпусах. Классификатор — по-прежнему логистическая регрессия,
поэтому результат объясним, а в расширении он вычисляется теми же несколькими арифметическими операциями."""
import json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from stylometry import stylometric_features, FEATURE_NAMES as STYLO
from lm_features import aggregate, aggregate_ext, LM_FEATURE_NAMES as LM, LM_EXTRA_NAMES as EXTRA

SETS = {'stylometric': STYLO, 'hybrid_v1': STYLO + LM, 'hybrid_v2': STYLO + LM + EXTRA}
CS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)


def feature_table(texts, langs, stats):
    rows = []
    for t, l, s in zip(texts, langs, stats):
        f = stylometric_features(t, l); f.update(aggregate(s)); f.update(aggregate_ext(s)); rows.append(f)
    return pd.DataFrame(rows)


def fit(X, y, C=1.0, sample_weight=None):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight='balanced'))
    if sample_weight is None:
        return m.fit(X, y)
    return m.fit(X, y, logisticregression__sample_weight=sample_weight)


def select_c(Xtr, ytr, val_sets, wtr=None, Cs=CS):
    """Подбор коэффициента регуляризации C по среднему AUROC на валидационных выборках val_sets = [(X, y), ...]."""
    best = None
    for C in Cs:
        m = fit(Xtr, ytr, C, wtr)
        score = float(np.mean([roc_auc_score(y, m.predict_proba(X)[:, 1]) for X, y in val_sets]))
        if best is None or score > best[1]:
            best = (C, score, m)
    return best   # (C, средний AUROC на валидации, модель)


def corpus_weights(groups):
    """Веса примеров: суммарный вес каждого корпуса одинаков."""
    groups = np.asarray(groups); w = np.ones(len(groups))
    for g in np.unique(groups):
        w[groups == g] = len(groups) / (len(np.unique(groups)) * (groups == g).sum())
    return w


def threshold_at_fpr(model, X_val, y_val, fpr):
    scores = model.predict_proba(X_val)[:, 1][np.asarray(y_val) == 0]
    return float(np.quantile(scores, 1 - fpr))


def model_json(model, cols, thresholds, C):
    sc, lr = model[0], model[-1]
    return {'features': list(cols), 'mean': sc.mean_.tolist(), 'scale': sc.scale_.tolist(),
            'coef': lr.coef_[0].tolist(), 'intercept': float(lr.intercept_[0]), 'thresholds': thresholds, 'C': C}


def export(langs, path, meta):
    """langs = {lang: {'main': dict, 'stage1': dict}} — основной классификатор и быстрый стилометрический (первая ступень)."""
    res = {'version': 'hybrid-v2', 'demo': False, 'meta': meta, 'langs': {}}
    for lang, d in langs.items():
        res['langs'][lang] = dict(d['main']); res['langs'][lang]['stage1'] = d['stage1']
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
