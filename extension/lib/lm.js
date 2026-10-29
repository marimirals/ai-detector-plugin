// Вероятностные признаки по компактной языковой модели — копия логики research/lm_features.py.
// Токенизатор — Transformers.js, сама модель (ONNX, INT8) — ONNX Runtime Web.
// Модель версии 2 (шаг 6) сама вычисляет для каждого токена 4 числа (log p, H, дисперсию, ранг) — выход «stats».
// Модель версии 1 возвращает логиты (50 тысяч чисел на токен), и величины считаются здесь, в tokenStats.
export const LM_BASE_NAMES = ['mean_lp', 'std_lp', 'mean_entropy', 'mean_log_rank',
  'rank_1', 'rank_2_10', 'rank_11_100', 'rank_101_1000', 'rank_gt_1000', 'curvature'];
export const LM_EXTRA_NAMES = ['lp_q10', 'lp_q50', 'lp_q90', 'std_entropy', 'mean_lp_plus_h',
  'xent_ratio', 'frac_above_expected', 'mean_var'];
export const LM_FEATURE_NAMES = [...LM_BASE_NAMES, ...LM_EXTRA_NAMES];

// Величины по токенам из логитов (модель версии 1): logits — Float32Array размера T×V
export function tokenStats(ids, logits, V) {
  const T = ids.length;
  const lp = [], ent = [], vr = [], rank = [];
  for (let t = 0; t < T - 1; t++) {
    const off = t * V;
    let max = -Infinity;
    for (let v = 0; v < V; v++) if (logits[off + v] > max) max = logits[off + v];
    let sum = 0;
    for (let v = 0; v < V; v++) sum += Math.exp(logits[off + v] - max);
    const lse = max + Math.log(sum);
    const target = Number(ids[t + 1]);
    const lpT = logits[off + target] - lse;
    let H = 0, E2 = 0, r = 1;
    for (let v = 0; v < V; v++) {
      const l = logits[off + v] - lse;
      const p = Math.exp(l);
      H -= p * l; E2 += p * l * l;
      if (l > lpT) r++;
    }
    lp.push(lpT); ent.push(H); vr.push(E2 - H * H); rank.push(r);
  }
  return { lp, ent, var: vr, rank };
}

// Величины по токенам из выхода «stats» (модель версии 2): data — Float32Array размера (T−1)×4
export function statsFromGraph(data, n) {
  const lp = new Array(n), ent = new Array(n), vr = new Array(n), rank = new Array(n);
  for (let i = 0; i < n; i++) {
    lp[i] = data[4 * i]; ent[i] = data[4 * i + 1]; vr[i] = data[4 * i + 2]; rank[i] = Math.round(data[4 * i + 3]);
  }
  return { lp, ent, var: vr, rank };
}

const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
const std = (a) => { const m = mean(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / a.length); };
// Квантиль с линейной интерполяцией — как numpy.quantile по умолчанию
export function quantile(a, q) {
  const s = [...a].sort((x, y) => x - y);
  const pos = q * (s.length - 1), lo = Math.floor(pos), hi = Math.ceil(pos);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}

export function aggregate(s) {
  const n = s.lp.length;
  if (!n) return Object.fromEntries(LM_FEATURE_NAMES.map((f) => [f, 0]));
  const m = mean(s.lp);
  const frac = (f) => s.rank.filter(f).length / n;
  const sumLp = s.lp.reduce((a, b) => a + b, 0);
  const sumH = s.ent.reduce((a, b) => a + b, 0);
  const sumVar = s.var.reduce((a, b) => a + b, 0);
  const meanH = sumH / n;
  let above = 0;
  for (let i = 0; i < n; i++) if (s.lp[i] > -s.ent[i]) above++;
  return {
    mean_lp: m, std_lp: std(s.lp), mean_entropy: meanH,
    mean_log_rank: mean(s.rank.map(Math.log)),
    rank_1: frac((r) => r === 1), rank_2_10: frac((r) => r >= 2 && r <= 10),
    rank_11_100: frac((r) => r >= 11 && r <= 100), rank_101_1000: frac((r) => r >= 101 && r <= 1000),
    rank_gt_1000: frac((r) => r > 1000),
    curvature: (sumLp + sumH) / Math.sqrt(Math.max(sumVar, 1e-8)),
    lp_q10: quantile(s.lp, 0.1), lp_q50: quantile(s.lp, 0.5), lp_q90: quantile(s.lp, 0.9),
    std_entropy: std(s.ent), mean_lp_plus_h: (sumLp + sumH) / n,
    xent_ratio: (-sumLp / n) / Math.max(meanH, 1e-6),
    frac_above_expected: above / n, mean_var: sumVar / n,
  };
}

// Загрузка токенизатора и модели. tokenizerLib — модуль Transformers.js, ort — модуль ONNX Runtime Web.
export async function loadLM({ tokenizerLib, ort, modelsUrl, name, file = 'model_quantized.onnx', modelBytes = null,
                             providers = ['wasm'] }) {
  const t0 = performance.now();
  tokenizerLib.env.allowRemoteModels = false;
  tokenizerLib.env.allowLocalModels = true;
  tokenizerLib.env.localModelPath = modelsUrl;
  const tokenizer = await tokenizerLib.AutoTokenizer.from_pretrained(name);
  const source = modelBytes ?? `${modelsUrl}${name}/onnx/${file}`;   // в Node.js (тесты) модель передаётся байтами
  const session = await ort.InferenceSession.create(source, { executionProviders: providers });
  const graphStats = session.outputNames.includes('stats');
  return { tokenizer, session, ort, graphStats, loadMs: performance.now() - t0 };
}

export function tokenize(lm, text, maxTokens = 256) {
  const enc = lm.tokenizer(text, { add_special_tokens: false, truncation: true, max_length: maxTokens });
  return Array.from(enc.input_ids.data, Number);
}

export async function lmFeatures(lm, text, maxTokens = 256) {
  const ids = tokenize(lm, text, maxTokens);
  if (ids.length < 2) return { ...aggregate({ lp: [], ent: [], var: [], rank: [] }), _tokens: ids.length };
  const T = ids.length;
  const toTensor = (arr) => new lm.ort.Tensor('int64', BigInt64Array.from(arr.map(BigInt)), [1, T]);
  const feeds = { input_ids: toTensor(ids), attention_mask: toTensor(new Array(T).fill(1)) };
  const out = await lm.session.run(feeds);
  const f = lm.graphStats ? aggregate(statsFromGraph(out.stats.data, T - 1))
    : aggregate(tokenStats(ids, out.logits.data, out.logits.dims[2]));
  f._tokens = T;                // число токенов фрагмента — по нему выбирается классификатор под длину
  return f;
}

// Текст, восстановленный из первых maxTokens токенов (для стилометрии в режиме «вся статья»)
export function truncateText(lm, text, maxTokens) {
  const ids = tokenize(lm, text, maxTokens);
  return lm.tokenizer.decode(ids, { skip_special_tokens: true, clean_up_tokenization_spaces: false });
}
