// Видимые настройки: пороги и минимальная длина абзаца. Быстрая предварительная оценка отключена,
// вычисления — всегда на процессоре (WebAssembly) с автоматическим числом потоков.
const DEFAULTS = { low: 0.4, high: 0.7, minWords: 40, modelThresholds: true, twoStage: false, backend: 'wasm', threads: 0 };
const $ = (id) => document.getElementById(id);
let saved = { ...DEFAULTS };

async function load() {
  saved = { ...DEFAULTS, ...(await chrome.storage.sync.get(DEFAULTS)) };
  $('low').value = Math.round(saved.low * 100); $('high').value = Math.round(saved.high * 100);
  $('minWords').value = saved.minWords; $('modelThresholds').checked = saved.modelThresholds;
}
function read() {
  return { low: Number($('low').value) / 100, high: Number($('high').value) / 100, minWords: Number($('minWords').value),
           modelThresholds: $('modelThresholds').checked, twoStage: false, backend: 'wasm', threads: 0 };
}
async function activeTab() { const [tab] = await chrome.tabs.query({ active: true, currentWindow: true }); return tab; }
async function send(msg) {
  try { const r = await chrome.tabs.sendMessage((await activeTab()).id, msg); $('msg').textContent = ''; return r ?? true; }
  catch { $('msg').textContent = 'На этой странице расширение не работает (служебная страница или страница не перезагружена после установки).'; return null; }
}
async function persist() {
  const s = read();
  // раньше можно было выбрать видеокарту или число потоков — при смене сбрасываем движок
  if (s.backend !== saved.backend || s.threads !== saved.threads) await chrome.runtime.sendMessage({ type: 'reset-engine' });
  await chrome.storage.sync.set(s); saved = s; return s;
}

// Кнопка «Убрать / Вернуть подсветку»: подсветка прячется, результаты остаются в памяти страницы и возвращаются без пересчёта
function setToggle(state) {
  const restore = !!state?.hidden;
  $('toggle').classList.toggle('restore', restore);
  $('toggleText').textContent = restore ? 'Вернуть подсветку' : 'Убрать подсветку';
  $('toggle').disabled = !state || (!state.active && !state.hidden);
}
async function refreshState() { setToggle(await send({ type: 'status' })); }

$('run').onclick = async () => { const s = await persist(); if (await send({ type: 'run', settings: s })) setToggle({ active: true, hidden: false }); };
$('article').onclick = async () => { const s = await persist(); send({ type: 'article', settings: s }); };
$('toggle').onclick = async () => {
  const restore = $('toggle').classList.contains('restore');
  setToggle(await send({ type: restore ? 'show' : 'hide' }));
};

$('settingsToggle').onclick = () => {
  const open = $('settingsToggle').getAttribute('aria-expanded') !== 'true';
  $('settingsToggle').setAttribute('aria-expanded', String(open)); $('settingsPanel').hidden = !open;
};
['low', 'high', 'minWords', 'modelThresholds'].forEach((id) => $(id).addEventListener('change', persist));
$('bench').onclick = (e) => { e.preventDefault(); chrome.tabs.create({ url: chrome.runtime.getURL('bench/bench.html') }); };

load().then(refreshState);
