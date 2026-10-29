// Шаг 6: проверка и замер кода расширения в Node.js (ONNX Runtime Web, бэкенд WebAssembly).
// Аргументы: входной JSON {lang: {model, texts}}, выходной JSON, папка с моделями, имя файла модели.
// ORT_THREADS — число потоков WebAssembly (по умолчанию 1).
import fs from 'fs';
import { normalize, stylometricFeatures } from './lib/features.js';
import { loadLM, lmFeatures, tokenize } from './lib/lm.js';

const [inPath, outPath, modelsDir, modelFile = 'model_quantized.onnx'] = process.argv.slice(2);
const threads = Number(process.env.ORT_THREADS || 1);
const ort = await import('onnxruntime-web');
ort.env.wasm.numThreads = threads;
const { AutoTokenizer, env } = await import('@huggingface/transformers');
const input = JSON.parse(fs.readFileSync(inPath, 'utf8'));
const result = { _threads: threads, _file: modelFile };
for (const [lang, { model, texts }] of Object.entries(input)) {
  const bytes = new Uint8Array(fs.readFileSync(`${modelsDir}/${model}/onnx/${modelFile}`));
  const lm = await loadLM({ tokenizerLib: { AutoTokenizer, env }, ort, modelsUrl: modelsDir + '/', name: model, modelBytes: bytes });
  console.log(`${lang}: модель ${modelFile} загружена за ${Math.round(lm.loadMs)} мс, выход: ${lm.session.outputNames.join(',')}, потоков: ${threads}`);
  await lmFeatures(lm, 'warm up text for the model');            // прогрев
  result[lang] = [];
  for (const t of texts) {
    const n = normalize(t);
    const t0 = performance.now();
    const f = { ...stylometricFeatures(n, lang), ...(await lmFeatures(lm, n, 256)) };
    f._ms = performance.now() - t0;
    f._ids = tokenize(lm, n, 256);
    result[lang].push(f);
    if (result[lang].length % 25 === 0) console.log(`  ${lang}: ${result[lang].length} из ${texts.length}`);
  }
}
fs.writeFileSync(outPath, JSON.stringify(result));
console.log('JS: готово');
