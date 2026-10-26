// Проверка в Node.js: признаки, вычисленные кодом расширения (JavaScript + Transformers.js + ONNX Runtime Web),
// совпадают с признаками, вычисленными в Python (ONNX Runtime). Запускается из ноутбука шага 3.
import fs from 'fs';
import { normalize, stylometricFeatures } from './lib/features.js';
import { loadLM, lmFeatures } from './lib/lm.js';

const [inPath, outPath, modelsDir, modelFile = 'model_quantized.onnx'] = process.argv.slice(2);
// В браузере расширение использует onnxruntime-web; если в Node.js он недоступен, берём onnxruntime-node
// (тот же движок ONNX Runtime, другая сборка) — для проверки кода признаков этого достаточно.
// ORT_BACKEND=node — принудительно onnxruntime-node (для диагностики: отличить ошибку в коде признаков от различий движков)
let ort, ortName;
try {
  if (process.env.ORT_BACKEND === 'node') throw new Error('выбран onnxruntime-node');
  ort = await import('onnxruntime-web'); ortName = 'onnxruntime-web';
}
catch (e) { console.log('onnxruntime-web не загрузился:', e.message); ort = await import('onnxruntime-node'); ortName = 'onnxruntime-node'; }
console.log('ONNX Runtime для теста:', ortName);
const { AutoTokenizer, env } = await import('@huggingface/transformers');
const input = JSON.parse(fs.readFileSync(inPath, 'utf8'));   // {lang: {model: name, texts: [...]}}
if (ort.env?.wasm) ort.env.wasm.numThreads = 1;
const result = {};
for (const [lang, { model, texts }] of Object.entries(input)) {
  const bytes = new Uint8Array(fs.readFileSync(`${modelsDir}/${model}/onnx/${modelFile}`));
  const lm = await loadLM({ tokenizerLib: { AutoTokenizer, env }, ort, modelsUrl: modelsDir + '/', name: model, modelBytes: bytes,
                            providers: ortName === 'onnxruntime-web' ? ['wasm'] : ['cpu'] });
  result[lang] = [];
  for (const t of texts) {
    const n = normalize(t);
    const t0 = performance.now();
    const f = { ...stylometricFeatures(n, lang), ...(await lmFeatures(lm, n, 256)) };
    f._ms = performance.now() - t0;
    f._norm = n;
    f._ids = Array.from(lm.tokenizer(n, { add_special_tokens: false, truncation: true, max_length: 256 }).input_ids.data, Number);
    result[lang].push(f);
  }
}
result._ort = ortName;
fs.writeFileSync(outPath, JSON.stringify(result));
console.log('JS: готово', Object.fromEntries(Object.entries(result).map(([k, v]) => [k, v.length])));
