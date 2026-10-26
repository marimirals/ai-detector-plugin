"""Стилометрические признаки (та же логика будет реализована в расширении на JavaScript)."""
import re, math, unicodedata
from collections import Counter

RU_FUNC = set("и в во не на я что он с со как а то все она так его но да ты к ко у же вы за бы по только ее её мне было вот от меня еще ещё нет о об из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой совсем ним здесь этом один почти мой тем чтобы нее неё сейчас были куда зачем всех никогда можно при наконец два об другой хоть после над больше тот через эти нас про всего них какая много разве три эту моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им более всегда конечно всю между".split())
EN_FUNC = set("the of and to a in is that it for on as with was be by this are or from at an but not have has had which they you he she we his her their its can will would there been were if all so no one more also do than about into such these other only some may then".split())

CYR2LAT = {"А":"A","В":"B","Е":"E","К":"K","М":"M","Н":"H","О":"O","Р":"P","С":"C","Т":"T","Х":"X",
           "а":"a","е":"e","о":"o","р":"p","с":"c","у":"y","х":"x"}
LAT2CYR = {v: k for k, v in CYR2LAT.items()}

def cyr_ratio(text):
    letters = [ch for ch in text if ch.isalpha()]
    if not letters: return 0.0
    return sum('Ѐ' <= ch <= 'ӿ' for ch in letters) / len(letters)

def latin_ratio(text):
    letters = [ch for ch in text if ch.isalpha()]
    if not letters: return 0.0
    return sum('a' <= ch.lower() <= 'z' for ch in letters) / len(letters)

def detect_lang(text):
    """'ru' — преобладает кириллица, 'en' — почти только латиница, иначе 'other' (не поддерживается)."""
    if cyr_ratio(text) > 0.5: return 'ru'
    if latin_ratio(text) > 0.9: return 'en'
    return 'other'

def _convert(word, table):
    return ''.join(table.get(ch, ch) for ch in word)

def _mixed(word):
    return any('\u0400' <= ch <= '\u04FF' for ch in word) and any('a' <= ch.lower() <= 'z' for ch in word)

def _fix_word(word, doc_script):
    """Слово из смеси алфавитов приводим к основному алфавиту текста; если это невозможно
    (в слове есть буквы без двойников) — к алфавиту большинства букв слова (защита от гомоглифов)."""
    if not _mixed(word):
        # слово целиком из «чужого» алфавита, но только из букв-двойников (например, кириллическая «а» в английском тексте)
        foreign = CYR2LAT if doc_script == 'lat' else LAT2CYR
        if word and all(ch in foreign for ch in word):
            return _convert(word, foreign)
        return word
    first = _convert(word, LAT2CYR if doc_script == 'cyr' else CYR2LAT)
    if not _mixed(first):
        return first
    cyr = sum('\u0400' <= ch <= '\u04FF' for ch in word)
    lat = sum('a' <= ch.lower() <= 'z' for ch in word)
    return _convert(word, LAT2CYR if cyr > lat else CYR2LAT)

def normalize(text):
    """NFKC, удаление невидимых символов, нормализация пробелов, исправление гомоглифов."""
    t = unicodedata.normalize('NFKC', text)
    t = re.sub(r'[\u200b\u200c\u200d\u2060\ufeff]', '', t)
    t = re.sub(r'[ \t\u00a0]+', ' ', t)
    t = re.sub(r'\s*\n\s*', '\n', t).strip()
    doc_script = 'cyr' if cyr_ratio(t) > 0.5 else 'lat'
    return re.sub(r'\w+', lambda m: _fix_word(m.group(0), doc_script), t)

SENT_SPLIT = re.compile(r'(?<=[.!?…])\s+')
WORD = re.compile(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*", re.UNICODE)

FEATURE_NAMES = ['mean_sent_len', 'cv_sent_len', 'mean_word_len', 'ttr', 'punct_ratio',
                 'func_ratio', 'rep_bigram_ratio', 'rep_trigram_ratio', 'comma_per_sent']

def stylometric_features(text, lang=None):
    lang = lang or detect_lang(text)
    sents = [s for s in SENT_SPLIT.split(text) if s.strip()]
    words = [w.lower() for w in WORD.findall(text)]
    n = max(len(words), 1)
    slens = [len(WORD.findall(s)) for s in sents] or [0]
    mu = sum(slens) / len(slens)
    sd = math.sqrt(sum((x - mu) ** 2 for x in slens) / len(slens))
    func = RU_FUNC if lang == 'ru' else EN_FUNC
    def rep(k):
        grams = [tuple(words[i:i+k]) for i in range(len(words) - k + 1)]
        if not grams: return 0.0
        c = Counter(grams); return sum(v for v in c.values() if v > 1) / len(grams)
    chars = max(len(text), 1)
    return {
        'mean_sent_len': mu,
        'cv_sent_len': sd / mu if mu else 0.0,
        'mean_word_len': sum(map(len, words)) / n,
        'ttr': len(set(words)) / n,
        'punct_ratio': sum(unicodedata.category(ch).startswith('P') for ch in text) / chars,
        'func_ratio': sum(w in func for w in words) / n,
        'rep_bigram_ratio': rep(2),
        'rep_trigram_ratio': rep(3),
        'comma_per_sent': text.count(',') / max(len(sents), 1),
    }
