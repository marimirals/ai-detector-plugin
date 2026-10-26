const DEFAULTS = { low: 0.4, high: 0.7, minWords: 40, modelThresholds: true };
const $ = (id) => document.getElementById(id);

async function load() {
  const s = { ...DEFAULTS, ...(await chrome.storage.sync.get(DEFAULTS)) };
  $('low').value = Math.round(s.low * 100); $('high').value = Math.round(s.high * 100); $('minWords').value = s.minWords; $('modelThresholds').checked = s.modelThresholds;
}
function read() {
  return { low: Number($('low').value) / 100, high: Number($('high').value) / 100, minWords: Number($('minWords').value), modelThresholds: $('modelThresholds').checked };
}
async function send(msg) {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  try { await chrome.tabs.sendMessage(tab.id, msg); $('msg').textContent = ''; }
  catch { $('msg').textContent = 'На этой странице расширение не работает (служебная страница или страница не перезагружена после установки).'; }
}
$('run').onclick = async () => { const s = read(); await chrome.storage.sync.set(s); send({ type: 'run', settings: s }); };
$('clear').onclick = () => send({ type: 'clear' });
load();
