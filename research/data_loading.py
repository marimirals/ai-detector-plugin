"""Поиск и загрузка данных M4 и RuATD в единую таблицу:
text | label (1 = ИИ, 0 = человек) | lang | source | generator | pair_id"""
import os, json, glob, hashlib
import pandas as pd
from stylometry import detect_lang

HUMAN_KEYS = ['human_text', 'human', 'text_human', 'original_text']
MACHINE_KEYS = ['machine_text', 'machine', 'text_machine', 'generated_text', 'ai_text']
TEXT_KEYS = ['text', 'Text', 'content']
LABEL_KEYS = ['label', 'Label', 'Class', 'class', 'labels']
MODEL_KEYS = ['model', 'Model', 'generator', 'source_model']

def _first(d, keys):
    for k in keys:
        if k in d and d[k] not in (None, ''):
            return k
    return None

def _read_records(path, limit=None):
    recs = []
    with open(path, encoding='utf-8') as f:
        head = f.read(1); f.seek(0)
        if head == '[':                       # обычный JSON-массив
            data = json.load(f); recs = data[:limit] if limit else data
        else:                                 # JSON Lines
            for line in f:
                line = line.strip()
                if not line: continue
                try: recs.append(json.loads(line))
                except json.JSONDecodeError: continue
                if limit and len(recs) >= limit: break
    return recs

def _pid(*parts):
    return hashlib.md5('|'.join(map(str, parts)).encode()).hexdigest()[:12]

def _label_value(v):
    s = str(v).strip().lower()
    if s in ('1', 'm', 'machine', 'ai', 'generated', 'fake', 'true'): return 1
    if s in ('0', 'h', 'human', 'real', 'false'): return 0
    return None

def inspect_files(root, exts=('.jsonl', '.json', '.csv', '.tsv'), max_files=200):
    """Печатает список файлов данных и ключи первой записи — для диагностики."""
    files = [p for p in glob.glob(os.path.join(root, '**', '*'), recursive=True)
             if p.lower().endswith(exts) and os.path.getsize(p) > 0]
    print(f'{root}: найдено файлов данных — {len(files)}')
    for p in sorted(files)[:max_files]:
        size = os.path.getsize(p) / 1e6
        try:
            if p.lower().endswith(('.csv', '.tsv')):
                cols = list(pd.read_csv(p, sep='\t' if p.endswith('.tsv') else ',', nrows=2).columns)
            else:
                r = _read_records(p, limit=1); cols = list(r[0].keys()) if r else []
        except Exception as e:
            cols = [f'ошибка чтения: {e}']
        print(f'  {os.path.relpath(p, root)}  ({size:.1f} МБ)  поля: {cols[:12]}')
    return files

def load_json_dir(root, dataset_name, limit_per_file=None):
    rows = []
    for p in glob.glob(os.path.join(root, '**', '*.json*'), recursive=True):
        try: recs = _read_records(p, limit=limit_per_file)
        except Exception: continue
        fname = os.path.splitext(os.path.basename(p))[0]
        for i, r in enumerate(recs):
            if not isinstance(r, dict): continue
            gen = r.get(_first(r, MODEL_KEYS)) if _first(r, MODEL_KEYS) else fname
            hk, mk = _first(r, HUMAN_KEYS), _first(r, MACHINE_KEYS)
            if hk or mk:                                   # параллельный формат (как в M4)
                pid = _pid(dataset_name, r[hk]) if hk else _pid(dataset_name, fname, i)  # пара = один человеческий текст
                if hk: rows.append((r[hk], 0, fname, 'human', pid))
                if mk:
                    m = r[mk]
                    for j, t in enumerate(m if isinstance(m, list) else [m]):
                        rows.append((t, 1, fname, str(gen), pid))
                continue
            tk, lk = _first(r, TEXT_KEYS), _first(r, LABEL_KEYS)
            if tk and lk is not None and _label_value(r[lk]) is not None:
                rows.append((r[tk], _label_value(r[lk]), fname, str(gen) if _label_value(r[lk]) else 'human', _pid(dataset_name, fname, i)))
    return _finish(rows, dataset_name)

def load_csv_dir(root, dataset_name):
    rows = []
    for p in glob.glob(os.path.join(root, '**', '*.[ct]sv'), recursive=True):
        try: df = pd.read_csv(p, sep='\t' if p.endswith('.tsv') else ',')
        except Exception: continue
        tk = _first({c: 1 for c in df.columns}, TEXT_KEYS)
        lk = _first({c: 1 for c in df.columns}, LABEL_KEYS)
        if not tk or not lk: continue
        fname = os.path.splitext(os.path.basename(p))[0]
        for i, (t, l) in enumerate(zip(df[tk], df[lk])):
            lab = _label_value(l)
            if lab is None:                  # многоклассовая разметка: 'Human' = 0, имя модели = 1
                lab = 0 if str(l).strip().lower() == 'human' else 1
                gen = 'human' if lab == 0 else str(l)
            else:
                gen = 'human' if lab == 0 else 'unknown'
            rows.append((t, lab, fname, gen, _pid(dataset_name, fname, i)))
    return _finish(rows, dataset_name)

# Язык M4 определяем по имени файла: кириллицей пишутся и болгарский, и казахский,
# латиницей — немецкий и индонезийский, поэтому автоопределение по алфавиту здесь ненадёжно.
M4_LANG_BY_PREFIX = {'russian': 'ru', 'arxiv': 'en', 'peerread': 'en', 'reddit': 'en', 'wikihow': 'en', 'wikipedia': 'en'}
BLOOMZ_HUMAN_KEYS = {'arxiv': 'abstract', 'wikihow': 'text', 'wikipedia': 'text', 'reddit': 'text'}

def load_m4(root):
    """Загрузка английской и русской частей M4 с учётом разных форматов файлов."""
    rows = []
    for p in sorted(glob.glob(os.path.join(root, 'data', '*.jsonl'))):
        fname = os.path.splitext(os.path.basename(p))[0]
        prefix, gen_from_name = fname.split('_')[0], fname.split('_')[-1]
        lang = M4_LANG_BY_PREFIX.get(prefix)
        if lang is None:
            continue
        for i, r in enumerate(_read_records(p)):
            gen = str(r.get('model') or gen_from_name)
            human = r.get('human_text') or (r.get(BLOOMZ_HUMAN_KEYS.get(prefix, '')) if 'bloomz' in fname else None)
            machine = r.get('machine_text')
            if prefix == 'peerread' and 'bloomz' in fname:          # отзывы хранятся списками
                human = (r.get('human_reviews') or [None])[0]
                machine = (r.get('bloom_reviews') or [None])[0]
                gen = 'bloomz'
            pid = _pid('M4', human) if isinstance(human, str) else _pid('M4', fname, i)
            if isinstance(human, str): rows.append((human, 0, fname, 'human', pid, lang))
            for t in (machine if isinstance(machine, list) else [machine]):
                if isinstance(t, str): rows.append((t, 1, fname, gen, pid, lang))
    df = pd.DataFrame(rows, columns=['text', 'label', 'source', 'generator', 'pair_id', 'lang'])
    df = df[df.text.str.strip().str.len() > 0]
    df['label'] = df.label.astype(int); df['dataset'] = 'M4'
    return df.drop_duplicates(['text', 'label']).reset_index(drop=True)

def load_coat(config='binary'):
    """Корпус CoAT (опубликованная версия RuATD) с Hugging Face. Столбцы определяются автоматически."""
    from datasets import load_dataset
    ds = load_dataset('RussianNLP/coat', config)
    print(ds)
    frames = []
    for split in ds:
        d = ds[split].to_pandas()
        feats = ds[split].features
        text_col = max([c for c in d.columns if d[c].dtype == object or pd.api.types.is_string_dtype(d[c])],
                       key=lambda c: d[c].astype(str).str.len().mean())
        label_col = None
        for c in d.columns:
            if c == text_col: continue
            names = getattr(feats[c], 'names', None)
            vals = names or sorted(map(str, d[c].dropna().unique()))[:50]
            if any(str(v).lower() in ('h', 'human', 'm', 'machine') for v in vals):
                label_col = c; break
            if c == 'label' and set(map(str, d[c].dropna().unique())) <= {'0', '1'}:
                label_col = c; break    # карточка CoAT: 1 — машинный текст, 0 — человеческий
        if label_col is None:
            print(f'[{split}] столбец с меткой не найден, столбцы: {list(d.columns)}'); continue
        names = getattr(feats[label_col], 'names', None)
        def to_name(v):
            if names and str(v).isdigit() and int(v) < len(names): return names[int(v)]
            return v
        lab = d[label_col].map(to_name)
        lab = lab.astype(str).str.lower()
        if lab.isin(['0', '1']).all():
            y = (lab == '1').astype(int); gen = ['human' if v == 0 else 'unknown' for v in y]
        elif lab.isin(['h', 'm', 'human', 'machine']).all():
            y = lab.isin(['m', 'machine']).astype(int); gen = ['human' if v == 0 else 'unknown' for v in y]
        else:                                     # многоклассовая разметка: human или имя модели
            y = (lab != 'human').astype(int); gen = lab.tolist()
        frames.append(pd.DataFrame({'text': d[text_col], 'label': y, 'source': f'coat_{split}', 'generator': gen,
                                    'pair_id': [_pid('coat', split, i) for i in range(len(d))], 'lang': 'ru'}))
        print(f'[{split}] текст: {text_col}, метка: {label_col}, примеры меток: {lab.unique()[:8]}')
    if not frames: return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True); df['dataset'] = 'RuATD'
    df = df[df.text.astype(str).str.strip().str.len() > 0]
    return df.drop_duplicates(['text', 'label']).reset_index(drop=True)

def _finish(rows, dataset_name):
    df = pd.DataFrame(rows, columns=['text', 'label', 'source', 'generator', 'pair_id'])
    if df.empty: return df
    df = df[df.text.apply(lambda t: isinstance(t, str) and len(t.strip()) > 0)].copy()
    df['lang'] = df.text.apply(detect_lang)
    df['label'] = df.label.astype(int)
    df['dataset'] = dataset_name
    return df.drop_duplicates(['text', 'label']).reset_index(drop=True)
