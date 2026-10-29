// Скрытый документ: держит Web Worker с моделью, пока расширение работает (раздел 2.2).
const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
const pending = new Map();
let nextId = 1;

worker.onmessage = (e) => {
  if (e.data?.type === 'log') { console.log('[AI detector]', e.data.text); return; }
  const cb = pending.get(e.data.requestId);
  if (cb) { pending.delete(e.data.requestId); cb(e.data); }
};

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg?.target !== 'offscreen' || msg.type !== 'analyze') return false;
  const requestId = nextId++;
  pending.set(requestId, sendResponse);
  worker.postMessage({ requestId, items: msg.items, minWords: msg.minWords, mode: msg.mode, config: msg.config });
  return true; // ответ придёт асинхронно
});
