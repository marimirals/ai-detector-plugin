// Страница замера производительности (раздел 3.6): загрузка моделей, прогрев, время анализа, память.
const $ = (id) => document.getElementById(id);
const log = (t) => { $('log').textContent += t + '\n'; $('log').scrollTop = 1e9; };

// Базовые абзацы для замера (длина важна, содержание — нет); повторяются, чтобы набрать 512 токенов
const BASE = {
  en: 'The small town library opened a new reading room last spring. Volunteers painted the walls, repaired old shelves and ' +
      'collected donated books from neighbours. On weekday evenings the room is now full of students who prepare for exams, ' +
      'retired engineers who discuss local history and parents who read aloud to their children. The librarian says that the ' +
      'most popular section is not fiction but practical guides about gardening, cooking and repairing bicycles. Next year the ' +
      'council plans to extend the opening hours and to organise a series of weekend lectures given by local residents. ',
  ru: 'Прошлой весной в небольшой городской библиотеке открылся новый читальный зал. Волонтёры покрасили стены, починили старые ' +
      'полки и собрали книги, которые принесли соседи. По вечерам в будни зал заполняют студенты, готовящиеся к экзаменам, ' +
      'инженеры на пенсии, обсуждающие историю города, и родители, читающие детям вслух. Библиотекарь рассказывает, что больше ' +
      'всего спрашивают не художественную литературу, а практические книги о садоводстве, кулинарии и ремонте велосипедов. ' +
      'В следующем году администрация собирается продлить часы работы и устроить цикл лекций, которые будут читать сами жители. ',
};
const TEXTS = Object.fromEntries(Object.entries(BASE).map(([l, t]) => [l, [0, 1, 2].map((k) => {
  const s = t.split('. '); return (s.slice(k).join('. ') + ' ' + t.repeat(6)).trim();   // немного разные тексты
})]));
const LENGTHS = [128, 256, 512];
const PER_LENGTH = 6;

const hc = navigator.hardwareConcurrency || 1;
$('env').textContent = `Ядер (логических): ${hc}; cross-origin isolation: ${self.crossOriginIsolated ? 'да' : 'нет'}; ` +
  `WebGPU: ${navigator.gpu ? 'есть' : 'нет'}; браузер: ${navigator.userAgent.match(/Chrome\/[\d.]+/)?.[0] || navigator.userAgent}`;

function configs() {
  const list = [{ backend: 'wasm', threads: 1 }];
  if (self.crossOriginIsolated) {
    for (const t of [2, 4, 8]) if (t <= hc) list.push({ backend: 'wasm', threads: t });
  }
  if (navigator.gpu) list.push({ backend: 'webgpu', threads: 1 });
  return list;
}

async function memoryMB() {
  try {
    if (performance.measureUserAgentSpecificMemory) {
      const m = await performance.measureUserAgentSpecificMemory();
      return Math.round(m.bytes / 2 ** 20);
    }
  } catch (e) { log('память: ' + e.message); }
  return null;
}

function runConfig(config) {
  return new Promise((resolve) => {
    const w = new Worker(new URL('../offscreen/worker.js', import.meta.url), { type: 'module' });
    w.onmessage = async (e) => {
      if (e.data.type === 'log' || e.data.type === 'progress') { log(e.data.text); return; }
      if (e.data.type === 'bench_result') {
        const mem = await memoryMB();
        w.terminate();
        resolve({ ...(e.data.result || { config, error: e.data.error }), memoryMB: mem });
      }
    };
    w.onerror = (e) => { w.terminate(); resolve({ config, error: e.message }); };
    w.postMessage({ type: 'bench', config, texts: TEXTS, lengths: LENGTHS, perLength: PER_LENGTH });
  });
}

function render(all) {
  const rows = [];
  for (const r of all) {
    const name = r.config.backend === 'webgpu' ? 'WebGPU' : `WASM, потоков: ${r.config.threads}`;
    if (r.error) { rows.push(`<tr><td>${name}</td><td colspan="8">ошибка: ${r.error.slice(0, 200)}</td></tr>`); continue; }
    for (const [lang, v] of Object.entries(r.langs)) {
      const b = v.byLength;
      rows.push(`<tr><td>${name}</td><td>${lang}</td><td>${Math.round(v.loadMs)}</td><td>${Math.round(v.warmupMs)}</td>` +
        LENGTHS.map((L) => `<td>${Math.round(b[L].lm_median)}</td>`).join('') +
        `<td>${v.stage1_ms.toFixed(1)}</td><td>${r.memoryMB ?? '—'}</td></tr>`);
    }
  }
  $('tables').innerHTML = `<table><tr><th>Режим</th><th>Язык</th><th>Загрузка, мс</th><th>Прогрев, мс</th>` +
    LENGTHS.map((L) => `<th>${L} токенов, мс</th>`).join('') + `<th>Стилометрия, мс</th><th>Память, МБ</th></tr>${rows.join('')}</table>`;
}

$('start').onclick = async () => {
  $('start').disabled = true; $('log').textContent = '';
  const all = [];
  for (const c of configs()) {
    log(`— режим ${c.backend}, потоков ${c.threads} —`);
    all.push(await runConfig(c)); render(all);
  }
  const report = { env: { hardwareConcurrency: hc, crossOriginIsolated: self.crossOriginIsolated, webgpu: !!navigator.gpu,
                          userAgent: navigator.userAgent, deviceMemoryGB: navigator.deviceMemory ?? null }, results: all };
  $('json').value = JSON.stringify(report);
  log('Готово.'); $('start').disabled = false;
};
$('copy').onclick = () => { $('json').select(); navigator.clipboard.writeText($('json').value); };
