// Web Worker: вычисления выполняются в фоновом потоке и не блокируют интерфейс (раздел 2.2).
// Режимы анализа: 'fast' — только стилометрия (первая ступень, мгновенно), 'full' — гибридный детектор.
// Вычислительный бэкенд: 'wasm' (WebAssembly, число потоков threads) или 'webgpu' (видеокарта, с запасным WASM).
import { normalize, detectLang, stylometricFeatures, wordCount } from '../lib/features.js';
import { classify } from '../lib/classifier.js';
import { loadLM, lmFeatures, truncateText, LM_FEATURE_NAMES } from '../lib/lm.js';

const MODELS_URL = new URL('../models/', import.meta.url).href;
const VENDOR_URL = new URL('../vendor/', import.meta.url).href;
let weights = null;
let engine = null;       // { config, libs: Promise, lms: {} }

function defaultThreads() {
  return self.crossOriginIsolated ? Math.max(1, Math.min(4, navigator.hardwareConcurrency || 1)) : 1;
}

async function loadWeights() {
  if (!weights) weights = await (await fetch(new URL('../model/weights.json', import.meta.url))).json();
  return weights;
}

function getEngine(config = {}) {
  const backend = config.backend === 'webgpu' && self.navigator?.gpu ? 'webgpu' : 'wasm';
  const threads = Number(config.threads) > 0 ? Number(config.threads) : defaultThreads();
  if (engine && engine.config.backend === backend && engine.config.threads === threads) return engine;
  if (engine) return engine;   // ONNX Runtime инициализируется один раз; смена настроек — перезапуском документа
  const libs = (async () => {
    const t0 = performance.now();
    const ort = await import(backend === 'webgpu' ? '../vendor/ort/ort.webgpu.min.mjs' : '../vendor/ort/ort.wasm.min.mjs');
    ort.env.wasm.wasmPaths = VENDOR_URL + 'ort/';
    ort.env.wasm.numThreads = self.crossOriginIsolated ? threads : 1;   // потоки требуют cross-origin isolation
    ort.env.wasm.proxy = false;
    const tokenizerLib = await import('../vendor/transformers/transformers.min.js');
    return { ort, tokenizerLib, importMs: performance.now() - t0 };
  })();
  engine = { config: { backend, threads: self.crossOriginIsolated ? threads : 1 }, libs, lms: {} };
  return engine;
}

function getLM(lang, eng) {
  if (!eng.lms[lang]) {
    eng.lms[lang] = (async () => {
      const { ort, tokenizerLib } = await eng.libs;
      const name = weights.meta?.browser_models?.[lang] || (lang === 'ru' ? 'rugpt3small' : 'gpt2');
      const providers = eng.config.backend === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'];
      const lm = await loadLM({ tokenizerLib, ort, modelsUrl: MODELS_URL, name, providers });
      const t0 = performance.now();
      await lmFeatures(lm, 'This is a short text to warm up the model. Это короткий текст для прогрева модели.');
      lm.warmupMs = performance.now() - t0;          // «прогрев» модели (раздел 4.1)
      self.postMessage({ type: 'log', text: `модель ${name}: загрузка ${Math.round(lm.loadMs)} мс, прогрев ${Math.round(lm.warmupMs)} мс, ${eng.config.backend}, потоков ${eng.config.threads}` });
      return lm;
    })();
  }
  return eng.lms[lang];
}

async function analyzeOne(text, mode, minWords, w, eng) {
  const norm = normalize(text);
  const lang = detectLang(norm);
  if (lang === 'other') return { status: 'unsupported_language' };
  if (wordCount(norm) < minWords) return { status: 'too_short' };
  const model = w.langs[lang];
  if (!model) return { status: 'no_model', lang };
  const f = stylometricFeatures(norm, lang);
  const needsLM = model.features.some((n) => LM_FEATURE_NAMES.includes(n));
  if (mode === 'fast' || !needsLM) {
    const m = (mode === 'fast' && model.stage1) ? model.stage1 : model;
    if (needsLM && m === model) return { status: 'skipped' };   // нет быстрой модели — ждём полной оценки
    const { p, top } = classify(f, m);
    return { status: 'ok', lang, p, top, thr: m.thresholds || null, stage: mode === 'fast' ? 1 : 2 };
  }
  const t1 = performance.now();
  const lm = await getLM(lang, eng);
  if (mode === 'article' && model.article) {
    // Вся статья: первые 512 токенов объединённого текста; стилометрия — по тексту этих токенов
    const L = model.article.max_tokens || 512;
    const fa = { ...stylometricFeatures(truncateText(lm, norm, L), lang), ...(await lmFeatures(lm, norm, L)) };
    const { p, top } = classify(fa, model.article);
    return { status: 'ok', lang, p, top, thr: model.article.thresholds || null, stage: 3, tokens: fa._tokens, lmMs: performance.now() - t1 };
  }
  Object.assign(f, await lmFeatures(lm, norm, w.meta?.max_tokens || 256));
  const lmMs = performance.now() - t1;
  const m = pickByLength(model, f._tokens);
  const { p, top } = classify(f, m);
  return { status: 'ok', lang, p, top, thr: m.thresholds || null, stage: 2, lmMs, tokens: f._tokens };
}

// Классификатор под длину фрагмента (шаг 9): by_length — список, упорядоченный по upto_tokens
function pickByLength(model, nTokens) {
  if (!model.by_length?.length) return model;
  return model.by_length.find((m) => nTokens <= m.upto_tokens) || model.by_length[model.by_length.length - 1];
}

// Замер производительности для страницы bench/bench.html (раздел 3.6)
async function bench({ config, texts, lengths, perLength }) {
  const w = await loadWeights();
  const eng = getEngine(config);
  const tImport = performance.now();
  const libs = await eng.libs;
  const out = { config: eng.config, crossOriginIsolated: self.crossOriginIsolated, importMs: libs.importMs, langs: {} };
  for (const lang of Object.keys(texts)) {
    const t0 = performance.now();
    const lm = await getLM(lang, eng);
    const r = { loadMs: lm.loadMs, warmupMs: lm.warmupMs, firstReadyMs: performance.now() - t0, providers: eng.config.backend, byLength: {} };
    for (const L of lengths) {
      const times = [], totals = [];
      for (let i = 0; i < perLength; i++) {
        const text = texts[lang][i % texts[lang].length];
        const t1 = performance.now();
        await lmFeatures(lm, text, L);
        times.push(performance.now() - t1);
        const t2 = performance.now();
        await analyzeOne(text, 'full', 1, { ...w, meta: { ...w.meta, max_tokens: L } }, eng);
        totals.push(performance.now() - t2);
      }
      const q = (a, p) => [...a].sort((x, y) => x - y)[Math.min(a.length - 1, Math.floor(p * a.length))];
      r.byLength[L] = { lm_median: q(times, 0.5), lm_p95: q(times, 0.95), total_median: q(totals, 0.5) };
      self.postMessage({ type: 'progress', text: `${eng.config.backend}/${eng.config.threads}: ${lang}, ${L} токенов — ${Math.round(r.byLength[L].lm_median)} мс` });
    }
    const t3 = performance.now();
    for (let i = 0; i < 20; i++) await analyzeOne(texts[lang][i % texts[lang].length], 'fast', 1, w, eng);
    r.stage1_ms = (performance.now() - t3) / 20;
    out.langs[lang] = r;
  }
  out.totalMs = performance.now() - tImport;
  return out;
}

self.onmessage = async (e) => {
  const msg = e.data;
  if (msg.type === 'bench') {
    try { self.postMessage({ type: 'bench_result', result: await bench(msg) }); }
    catch (err) { self.postMessage({ type: 'bench_result', error: String(err?.stack || err) }); }
    return;
  }
  const { requestId, items, minWords, mode = 'full', config } = msg;
  const w = await loadWeights();
  const eng = getEngine(config);
  const t0 = performance.now();
  const results = [];
  for (const { id, text } of items) {
    try { results.push({ id, ...(await analyzeOne(text, mode, minWords, w, eng)) }); }
    catch (err) { results.push({ id, status: 'error', error: String(err) }); }
  }
  self.postMessage({ requestId, results, demo: !!w.demo, ms: performance.now() - t0, engine: eng.config });
};
