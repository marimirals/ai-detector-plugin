"""Базовый стилометрический детектор: признаки + логистическая регрессия, отдельно для каждого языка."""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupShuffleSplit
from stylometry import stylometric_features, normalize, FEATURE_NAMES
from metrics_utils import evaluate

def _features_with_format(text, lang):
    t = normalize(text)
    f = stylometric_features(t, lang)
    f['newline_ratio'] = t.count('\n') / max(len(t), 1)   # признак форматирования — только для эксперимента-абляции
    return f

def featurize(df, names=FEATURE_NAMES):
    return np.array([[f[k] for k in names] for f in
                     (_features_with_format(t, l) for t, l in zip(df.text, df.lang))])

def split(df, seed=42):
    """70/15/15 с разбиением по pair_id: человеческий текст и его сгенерированная пара не разделяются."""
    g1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    tr, rest = next(g1.split(df, groups=df.pair_id))
    rest_df = df.iloc[rest]
    g2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    va, te = next(g2.split(rest_df, groups=rest_df.pair_id))
    return df.iloc[tr], rest_df.iloc[va], rest_df.iloc[te]

def run(df, lang, cap_per_class=20000, seed=42, names=FEATURE_NAMES):
    d = df[df.lang == lang]
    d = pd.concat([g.sample(min(len(g), cap_per_class), random_state=seed) for _, g in d.groupby('label')])
    if d.label.nunique() < 2:
        return None, None, None
    tr, va, te = split(d, seed)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced'))
    model.fit(featurize(tr, names), tr.label)
    res = evaluate(te.label.values, model.predict_proba(featurize(te, names))[:, 1])
    res.update({'lang': lang, 'train': len(tr), 'val': len(va), 'test': len(te)})
    coefs = pd.Series(model[-1].coef_[0], index=names).sort_values(key=abs, ascending=False)
    return res, coefs, model

def export_weights(models, path):
    """Сохраняет параметры нормировки и логистической регрессии в JSON для расширения (extension/model/weights.json)."""
    import json
    out = {'version': 'stylometry-v1', 'demo': False, 'langs': {}}
    for lang, m in models.items():
        sc, lr = m[0], m[-1]
        out['langs'][lang] = {'features': FEATURE_NAMES, 'mean': sc.mean_.tolist(), 'scale': sc.scale_.tolist(),
                              'coef': lr.coef_[0].tolist(), 'intercept': float(lr.intercept_[0])}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)

def balanced_sample(d, cap_per_class, seed=42):
    return pd.concat([g.sample(min(len(g), cap_per_class), random_state=seed) for _, g in d.groupby('label')])

def run_cross(train_df, test_df, lang, cap_per_class=20000, seed=42, names=FEATURE_NAMES):
    """Обучение на одном корпусе и проверка на другом (перенос на незнакомые генераторы, серия Э3)."""
    tr = balanced_sample(train_df[train_df.lang == lang], cap_per_class, seed)
    te = balanced_sample(test_df[test_df.lang == lang], cap_per_class // 4, seed)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced'))
    model.fit(featurize(tr, names), tr.label)
    res = evaluate(te.label.values, model.predict_proba(featurize(te, names))[:, 1])
    res.update({'lang': lang, 'train': len(tr), 'test': len(te)})
    return res
