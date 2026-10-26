// Вероятностные признаки по компактной языковой модели — копия логики research/lm_features.py.
// Токенизатор — Transformers.js, сама модель (ONNX, INT8) — ONNX Runtime Web (WASM).
export const LM_FEATURE_NAMES = ['mean_lp', 'std_lp', 'mean_entropy', 'mean_log_rank',
  'rank_1', 'rank_2_10', 'rank_11_100', 'rank_101_1000', 'rank_gt_1000', 'curvature'];

// Статистики по токенам из логитов модели: logits — Float32Array размера T×V
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

const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;

export function aggregate(s) {
  const n = s.lp.length;
  if (!n) return Object.fromEntries(LM_FEATURE_NAMES.map((f) => [f, 0]));
  const m = mean(s.lp);
  const std = Math.sqrt(s.lp.reduce((a, x) => a + (x - m) ** 2, 0) / n);
  const frac = (f) => s.rank.filter(f).length / n;
  const sumLp = s.lp.reduce((a, b) => a + b, 0);
  const sumH = s.ent.reduce((a, b) => a + b, 0);
  const sumVar = s.var.reduce((a, b) => a + b, 0);
  return {
    mean_lp: m, std_lp: std, mean_entropy: mean(s.ent),
    mean_log_rank: mean(s.rank.map(Math.log)),
    rank_1: frac((r) => r === 1), rank_2_10: frac((r) => r >= 2 && r <= 10),
    rank_11_100: frac((r) => r >= 11 && r <= 100), rank_101_1000: frac((r) => r >= 101 && r <= 1000),
    rank_gt_1000: frac((r) => r > 1000),
    curvature: (sumLp + sumH) / Math.sqrt(Math.max(sumVar, 1e-8)),
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
  return { tokenizer, session, ort, loadMs: performance.now() - t0 };
}

export async function lmFeatures(lm, text, maxTokens = 256) {
  const enc = lm.tokenizer(text, { add_special_tokens: false, truncation: true, max_length: maxTokens });
  const ids = Array.from(enc.input_ids.data, Number);
  if (ids.length < 2) return aggregate({ lp: [], ent: [], var: [], rank: [] });
  const T = ids.length;
  const toTensor = (arr) => new lm.ort.Tensor('int64', BigInt64Array.from(arr.map(BigInt)), [1, T]);
  const out = await lm.session.run({ input_ids: toTensor(ids), attention_mask: toTensor(new Array(T).fill(1)) });
  const logits = out.logits;
  const V = logits.dims[2];
  return aggregate(tokenStats(ids, logits.data, V));
}
