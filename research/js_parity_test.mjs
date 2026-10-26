// Проверка в Node.js: признаки, вычисленные кодом расширения (JavaScript + Transformers.js + ONNX Runtime Web),
// совпадают с признаками, вычисленными в Python (ONNX Runtime). Запускается из ноутбука шага 3.
import fs from 'fs';
import * as ort from 'onnxruntime-web';
import { AutoTokenizer, env } from '@huggingface/transformers';
import { normalize, stylometricFeatures } from './lib/features.js';
import { loadLM, lmFeatures } from './lib/lm.js';

const [inPath, outPath, modelsDir] = process.argv.slice(2);
const input = JSON.parse(fs.readFileSync(inPath, 'utf8'));   // {lang: {model: name, texts: [...]}}
ort.env.wasm.numThreads = 1;
const result = {};
for (const [lang, { model, texts }] of Object.entries(input)) {
  const bytes = fs.readFileSync(`${modelsDir}/${model}/onnx/model_quantized.onnx`);
  const lm = await loadLM({ tokenizerLib: { AutoTokenizer, env }, ort, modelsUrl: modelsDir + '/', name: model, modelBytes: bytes });
  result[lang] = [];
  for (const t of texts) {
    const n = normalize(t);
    const t0 = performance.now();
    const f = { ...stylometricFeatures(n, lang), ...(await lmFeatures(lm, n, 256)) };
    f._ms = performance.now() - t0;
    f._norm = n;
    result[lang].push(f);
  }
}
fs.writeFileSync(outPath, JSON.stringify(result));
console.log('JS: готово', Object.fromEntries(Object.entries(result).map(([k, v]) => [k, v.length])));
