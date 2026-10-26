// Web Worker: вычисления выполняются в фоновом потоке и не блокируют интерфейс (раздел 2.2).
import { normalize, detectLang, stylometricFeatures, wordCount } from '../lib/features.js';
import { classify } from '../lib/classifier.js';
import { loadLM, lmFeatures, LM_FEATURE_NAMES } from '../lib/lm.js';

const MODELS_URL = new URL('../models/', import.meta.url).href;
const VENDOR_URL = new URL('../vendor/', import.meta.url).href;
let weights = null;
const lms = {};          // язык -> Promise загруженной модели
let libs = null;

async function loadWeights() {
  if (!weights) weights = await (await fetch('../model/weights.json')).json();
  return weights;
}

async function loadLibs() {
  if (!libs) {
    libs = (async () => {
      const ort = await import('../vendor/ort/ort.wasm.min.mjs');
      ort.env.wasm.wasmPaths = VENDOR_URL + 'ort/';
      ort.env.wasm.numThreads = 1;      // многопоточность требует cross-origin isolation, в расширении её нет
      ort.env.wasm.proxy = false;
      const tokenizerLib = await import('../vendor/transformers/transformers.min.js');
      return { ort, tokenizerLib };
    })();
  }
  return libs;
}

function getLM(lang) {
  if (!lms[lang]) {
    lms[lang] = (async () => {
      const { ort, tokenizerLib } = await loadLibs();
      const name = weights.meta?.browser_models?.[lang] || (lang === 'ru' ? 'rugpt3small' : 'gpt2');
      const lm = await loadLM({ tokenizerLib, ort, modelsUrl: MODELS_URL, name });
      await lmFeatures(lm, 'warm up');      // «прогрев» модели (раздел 4.1)
      self.postMessage({ type: 'log', text: `модель ${name} загружена за ${Math.round(lm.loadMs)} мс` });
      return lm;
    })();
  }
  return lms[lang];
}

self.onmessage = async (e) => {
  const { requestId, items, minWords } = e.data;
  const w = await loadWeights();
  const t0 = performance.now();
  const results = [];
  for (const { id, text } of items) {
    try {
      const norm = normalize(text);
      const lang = detectLang(norm);
      if (lang === 'other') { results.push({ id, status: 'unsupported_language' }); continue; }
      if (wordCount(norm) < minWords) { results.push({ id, status: 'too_short' }); continue; }
      const model = w.langs[lang];
      if (!model) { results.push({ id, status: 'no_model', lang }); continue; }
      const f = stylometricFeatures(norm, lang);
      if (model.features.some((n) => LM_FEATURE_NAMES.includes(n))) {
        const t1 = performance.now();
        Object.assign(f, await lmFeatures(await getLM(lang), norm, w.meta?.max_tokens || 256));
        f._lm_ms = performance.now() - t1;
      }
      const { p, top } = classify(f, model);
      results.push({ id, status: 'ok', lang, p, top, thr: model.thresholds || null, lmMs: f._lm_ms });
    } catch (err) {
      results.push({ id, status: 'error', error: String(err) });
    }
  }
  self.postMessage({ requestId, results, demo: !!w.demo, ms: performance.now() - t0 });
};
