// Сценарий содержимого: извлечение текстовых блоков, ленивый анализ и подсветка (разделы 2.2, 2.5).
(() => {
  const BLOCK_SELECTOR = 'p, li, blockquote, td, dd, figcaption, h1, h2, h3';
  const ATTR = 'data-aid-id';
  const DEFAULTS = { low: 0.4, high: 0.7, minWords: 40, modelThresholds: true };
  let settings = { ...DEFAULTS };
  let active = false;
  let observer = null;
  let queue = new Map();
  let timer = null;
  let nextId = 1;
  const cache = new Map(); // id -> результат, чтобы не анализировать блок повторно

  function isVisibleBlock(el) {
    if (el.closest('input, textarea, [contenteditable="true"], nav, header, footer, script, style')) return false;
    if (el.querySelector(BLOCK_SELECTOR)) return false; // берём самые «внутренние» блоки
    const style = getComputedStyle(el);
    return style.display !== 'none' && style.visibility !== 'hidden';
  }

  function collectBlocks() {
    return [...document.querySelectorAll(BLOCK_SELECTOR)].filter(
      (el) => isVisibleBlock(el) && el.innerText.trim().split(/\s+/).length >= Math.min(settings.minWords, 15)
    );
  }

  // Пороги: по умолчанию — подобранные на валидации (≈10 % и ≈2 % ложных срабатываний), иначе — заданные пользователем
  function colorFor(p, thr) {
    const useModel = settings.modelThresholds !== false && thr;
    const high = useModel ? thr.fpr02 : settings.high;
    const low = useModel ? thr.fpr10 : settings.low;
    if (p >= high) return 'rgba(239, 68, 68, 0.28)';   // красный
    if (p >= low) return 'rgba(250, 204, 21, 0.35)';   // жёлтый
    return null;
  }

  function paint(el, r, demo) {
    el.removeAttribute('title');
    el.style.removeProperty('background-color');
    if (r.status !== 'ok') return;
    const color = colorFor(r.p, r.thr);
    const pct = Math.round(r.p * 100);
    const reasons = r.top.map((t) => `• ${t.label} — ${t.contribution > 0 ? 'похоже на ИИ' : 'похоже на человека'}`).join('\n');
    el.title = `Вероятность генерации ИИ: ${pct}%${demo ? ' (ДЕМО-веса!)' : ''}\n` +
      `Основные признаки:\n${reasons}\n\nОценка вероятностная и не является доказательством.`;
    if (color) el.style.setProperty('background-color', color, 'important');
  }

  function flush() {
    timer = null;
    if (!queue.size) return;
    const batch = [...queue.entries()].slice(0, 20);
    batch.forEach(([id]) => queue.delete(id));
    chrome.runtime.sendMessage(
      { type: 'analyze', minWords: settings.minWords, items: batch.map(([id, el]) => ({ id, text: el.innerText })) },
      (res) => {
        if (chrome.runtime.lastError || !res || res.error) {
          console.warn('[AI detector]', chrome.runtime.lastError?.message || res?.error);
          return;
        }
        for (const r of res.results) {
          cache.set(r.id, { ...r, demo: res.demo });
          const el = document.querySelector(`[${ATTR}="${r.id}"]`);
          if (el && active) paint(el, r, res.demo);
        }
        if (queue.size) schedule();
      }
    );
  }
  function schedule() { if (!timer) timer = setTimeout(flush, 250); }

  function start() {
    active = true;
    observer = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        const el = e.target;
        const id = el.getAttribute(ATTR);
        if (cache.has(id)) { const c = cache.get(id); paint(el, c, c.demo); continue; }
        queue.set(id, el); schedule();
      }
    }, { rootMargin: '200px' });
    for (const el of collectBlocks()) {
      if (!el.hasAttribute(ATTR)) el.setAttribute(ATTR, String(nextId++));
      observer.observe(el);
    }
  }

  function stop() {
    active = false;
    observer?.disconnect(); observer = null; queue.clear();
    document.querySelectorAll(`[${ATTR}]`).forEach((el) => {
      el.style.removeProperty('background-color'); el.removeAttribute('title');
    });
  }

  chrome.runtime.onMessage.addListener((msg, _s, sendResponse) => {
    if (msg?.type === 'run') { settings = { ...DEFAULTS, ...msg.settings }; stop(); cache.clear(); start(); }
    if (msg?.type === 'clear') stop();
    if (msg?.type === 'status') sendResponse({ active });
  });
})();
