"""Обучение и оценка статистического, стилометрического и гибридного детекторов (разделы 2.3, 3.3)."""
import json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from stylometry import stylometric_features, normalize, FEATURE_NAMES as STYLO
from lm_features import aggregate, LM_FEATURE_NAMES as LM
from metrics_utils import evaluate

VARIANTS = {'stylometric': STYLO, 'statistical': LM, 'hybrid': STYLO + LM}

def feature_table(texts, langs, stats, k=None, decode=None):
    """Признаки обеих групп. Если задан k, стилометрия считается по тексту первых k+1 токенов."""
    rows = []
    for t, l, s in zip(texts, langs, stats):
        if k is not None and decode is not None:
            t = decode(s['ids'][:k + 1], l)
        f = stylometric_features(t, l)
        f.update(aggregate(s, k))
        rows.append(f)
    return pd.DataFrame(rows)

def fit(X, y):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight='balanced'))
    return m.fit(X, y)

def threshold_at_fpr(model, X_val, y_val, fpr):
    """Порог, при котором на валидационной выборке доля ложных срабатываний равна fpr."""
    scores = model.predict_proba(X_val)[:, 1][np.asarray(y_val) == 0]
    return float(np.quantile(scores, 1 - fpr))

def train_eval_all(F_tr, y_tr, F_te, y_te, variants=VARIANTS):
    out, models = {}, {}
    for name, cols in variants.items():
        m = fit(F_tr[cols].values, y_tr)
        out[name] = evaluate(y_te, m.predict_proba(F_te[cols].values)[:, 1])
        models[name] = m
    return out, models

def export(models, thresholds, path, meta):
    """Веса гибридного детектора для расширения."""
    res = {'version': 'hybrid-v1', 'demo': False, 'meta': meta, 'langs': {}}
    for lang, m in models.items():
        sc, lr = m[0], m[-1]
        res['langs'][lang] = {'features': STYLO + LM, 'mean': sc.mean_.tolist(), 'scale': sc.scale_.tolist(),
                              'coef': lr.coef_[0].tolist(), 'intercept': float(lr.intercept_[0]),
                              'thresholds': thresholds[lang]}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1)
