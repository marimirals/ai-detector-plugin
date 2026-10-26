"""Простые атаки на детекторы (серия Э5): замена букв гомоглифами и внесение опечаток."""
import random, re
from stylometry import CYR2LAT, LAT2CYR

def homoglyph_attack(text, rate=0.15, seed=0):
    """Заменяет долю rate подходящих букв визуально похожими буквами другого алфавита."""
    rnd = random.Random(seed)
    table = {**CYR2LAT, **LAT2CYR}
    return ''.join(table[ch] if ch in table and rnd.random() < rate else ch for ch in text)

def typo_attack(text, rate=0.1, seed=0):
    """В доле rate слов длиной от 4 букв переставляет две соседние буквы."""
    rnd = random.Random(seed)
    def swap(m):
        w = m.group(0)
        if len(w) < 4 or rnd.random() >= rate: return w
        i = rnd.randrange(1, len(w) - 2)
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    return re.sub(r'\w+', swap, text)
