// Стилометрические признаки — точная копия логики research/stylometry.py (раздел 2.3 диплома).
import { RU_FUNC, EN_FUNC, CYR2LAT, FEATURE_NAMES } from './wordlists.js';
export { FEATURE_NAMES };

const LAT2CYR = Object.fromEntries(Object.entries(CYR2LAT).map(([k, v]) => [v, k]));
const isCyr = (ch) => ch >= 'Ѐ' && ch <= 'ӿ';
const isLatin = (ch) => { const c = ch.toLowerCase(); return c >= 'a' && c <= 'z'; };

export function cyrRatio(text) {
  const letters = [...text].filter((ch) => /\p{L}/u.test(ch));
  if (!letters.length) return 0;
  return letters.filter(isCyr).length / letters.length;
}

export function latinRatio(text) {
  const letters = [...text].filter((ch) => /\p{L}/u.test(ch));
  if (!letters.length) return 0;
  return letters.filter(isLatin).length / letters.length;
}

// 'ru' — преобладает кириллица, 'en' — почти только латиница, иначе 'other' (не поддерживается)
export function detectLang(text) {
  if (cyrRatio(text) > 0.5) return 'ru';
  if (latinRatio(text) > 0.9) return 'en';
  return 'other';
}

const convert = (word, table) => [...word].map((ch) => table[ch] ?? ch).join('');
const isMixed = (word) => [...word].some(isCyr) && [...word].some(isLatin);

// Слово из смеси алфавитов приводим к основному алфавиту текста; если это невозможно —
// к алфавиту большинства букв слова. Слово целиком из «чужих» букв-двойников тоже исправляем.
function fixWord(word, docScript) {
  if (!isMixed(word)) {
    const foreign = docScript === 'lat' ? CYR2LAT : LAT2CYR;
    if (word && [...word].every((ch) => ch in foreign)) return convert(word, foreign);
    return word;
  }
  const first = convert(word, docScript === 'cyr' ? LAT2CYR : CYR2LAT);
  if (!isMixed(first)) return first;
  const chars = [...word];
  const cyr = chars.filter(isCyr).length;
  const lat = chars.filter(isLatin).length;
  return convert(word, cyr > lat ? LAT2CYR : CYR2LAT);
}

// NFKC, удаление невидимых символов, нормализация пробелов, исправление гомоглифов
export function normalize(text) {
  let t = text.normalize('NFKC');
  t = t.replace(/[\u200b\u200c\u200d\u2060\ufeff]/g, '');
  t = t.replace(/[ \t\u00a0]+/g, ' ');
  t = t.replace(/\s*\n\s*/g, '\n').trim();
  const docScript = cyrRatio(t) > 0.5 ? 'cyr' : 'lat';
  return t.replace(/[\p{L}\p{N}_]+/gu, (w) => fixWord(w, docScript));
}

const SENT_SPLIT = /(?<=[.!?…])\s+/u;
const WORD = /\p{L}+(?:[-'’]\p{L}+)*/gu;

export function stylometricFeatures(text, lang) {
  lang = lang || detectLang(text);
  const sents = text.split(SENT_SPLIT).filter((s) => s.trim());
  const words = (text.match(WORD) || []).map((w) => w.toLowerCase());
  const n = Math.max(words.length, 1);
  let slens = sents.map((s) => (s.match(WORD) || []).length);
  if (!slens.length) slens = [0];
  const mu = slens.reduce((a, b) => a + b, 0) / slens.length;
  const sd = Math.sqrt(slens.reduce((a, x) => a + (x - mu) ** 2, 0) / slens.length);
  const func = lang === 'ru' ? RU_FUNC : EN_FUNC;
  const rep = (k) => {
    const total = words.length - k + 1;
    if (total <= 0) return 0;
    const counts = new Map();
    for (let i = 0; i < total; i++) {
      const g = words.slice(i, i + k).join('\u0001');
      counts.set(g, (counts.get(g) || 0) + 1);
    }
    let s = 0; for (const v of counts.values()) if (v > 1) s += v;
    return s / total;
  };
  const chars = Math.max([...text].length, 1);
  const punct = [...text].filter((ch) => /\p{P}/u.test(ch)).length;
  return {
    mean_sent_len: mu,
    cv_sent_len: mu ? sd / mu : 0,
    mean_word_len: words.reduce((a, w) => a + [...w].length, 0) / n,
    ttr: new Set(words).size / n,
    punct_ratio: punct / chars,
    func_ratio: words.filter((w) => func.has(w)).length / n,
    rep_bigram_ratio: rep(2),
    rep_trigram_ratio: rep(3),
    comma_per_sent: (text.split(',').length - 1) / Math.max(sents.length, 1),
  };
}

export function wordCount(text) { return (text.match(WORD) || []).length; }
