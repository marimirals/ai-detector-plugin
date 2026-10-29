const DEFAULTS = { low: 0.4, high: 0.7, minWords: 40, modelThresholds: true, twoStage: true, backend: 'wasm', threads: 0 };
const $ = (id) => document.getElementById(id);
let saved = { ...DEFAULTS };

async function load() {
  saved = { ...DEFAULTS, ...(await chrome.storage.sync.get(DEFAULTS)) };
  $('low').value = Math.round(saved.low * 100); $('high').value = Math.round(saved.high * 100);
  $('minWords').value = saved.minWords; $('modelThresholds').checked = saved.modelThresholds;
  $('twoStage').checked = saved.twoStage; $('backend').value = saved.backend; $('threads').value = saved.threads;
}
function read() {
  return { low: Number($('low').value) / 100, high: Number($('high').value) / 100, minWords: Number($('minWords').value),
           modelThresholds: $('modelThresholds').checked, twoStage: $('twoStage').checked,
           backend: $('backend').value, threads: Number($('threads').value) || 0 };
}
async function send(msg) {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  try { await chrome.tabs.sendMessage(tab.id, msg); $('msg').textContent = ''; }
  catch { $('msg').textContent = 'На этой странице расширение не работает (служебная страница или страница не перезагружена после установки).'; }
}
$('run').onclick = async () => {
  const s = read();
  if (s.backend !== saved.backend || s.threads !== saved.threads) await chrome.runtime.sendMessage({ type: 'reset-engine' });
  await chrome.storage.sync.set(s); saved = s;
  send({ type: 'run', settings: s });
};
$('clear').onclick = () => send({ type: 'clear' });
$('article').onclick = async () => { const s = read(); await chrome.storage.sync.set(s); saved = s; send({ type: 'article', settings: s }); };
$('bench').onclick = (e) => { e.preventDefault(); chrome.tabs.create({ url: chrome.runtime.getURL('bench/bench.html') }); };
load();
