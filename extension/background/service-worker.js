// Сервис-воркер: маршрутизация сообщений и управление offscreen-документом (раздел 2.2).
const OFFSCREEN_URL = 'offscreen/offscreen.html';
let creating = null;

async function hasOffscreen() {
  const contexts = await chrome.runtime.getContexts({
    contextTypes: ['OFFSCREEN_DOCUMENT'],
    documentUrls: [chrome.runtime.getURL(OFFSCREEN_URL)],
  });
  return contexts.length > 0;
}

async function ensureOffscreen() {
  if (await hasOffscreen()) return;
  if (!creating) {
    creating = chrome.offscreen.createDocument({
      url: OFFSCREEN_URL,
      reasons: ['WORKERS'],
      justification: 'Запуск модели детектирования в Web Worker',
    }).finally(() => { creating = null; });
  }
  await creating;
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  // Смена вычислительного бэкенда или числа потоков: документ с моделью создаётся заново при следующем анализе
  if (msg?.type === 'reset-engine') {
    (async () => { if (await hasOffscreen()) await chrome.offscreen.closeDocument(); sendResponse({ ok: true }); })();
    return true;
  }
  if (msg?.type !== 'analyze' || msg.target === 'offscreen') return false;
  (async () => {
    try {
      await ensureOffscreen();
      const res = await chrome.runtime.sendMessage({ ...msg, target: 'offscreen' });
      sendResponse(res);
    } catch (err) {
      sendResponse({ error: String(err) });
    }
  })();
  return true;
});
