// Сценарий содержимого: извлечение текстовых блоков, ленивый анализ и подсветка (разделы 2.2, 2.5).
(() => {
  const BLOCK_SELECTOR = 'p, li, blockquote, td, dd, figcaption, h1, h2, h3';
  const ATTR = 'data-aid-id';
  const DEFAULTS = { low: 0.4, high: 0.7, minWords: 40, modelThresholds: true, twoStage: true, backend: 'wasm', threads: 0 };
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

  // stage 1 — предварительная оценка по стилометрии (пунктирная рамка), stage 2 — гибридный детектор
  function paint(el, r, demo) {
    if (r.status !== 'ok') {
      if (r.status === 'error' && el.dataset.aidStage === '1') el.title = el.title.replace('уточняется…', 'уточнить не удалось');
      else if (r.status !== 'skipped' && !el.dataset.aidStage) clearEl(el);
      return;
    }
    if (r.stage === 1 && el.dataset.aidStage === '2') return;   // окончательная оценка уже есть
    clearEl(el);
    el.dataset.aidStage = String(r.stage || 2);
    const color = colorFor(r.p, r.thr);
    const pct = Math.round(r.p * 100);
    const reasons = r.top.map((t) => `• ${t.label} — ${t.contribution > 0 ? 'похоже на ИИ' : 'похоже на человека'}`).join('\n');
    const head = r.stage === 1 ? 'Предварительная оценка (только стиль текста), уточняется…' : 'Вероятность генерации ИИ';
    el.title = `${head}: ${pct}%${demo ? ' (ДЕМО-веса!)' : ''}\n` +
      `Основные признаки:\n${reasons}\n\nОценка вероятностная и не является доказательством.`;
    if (color) el.style.setProperty('background-color', color, 'important');
    if (r.stage === 1) el.style.setProperty('outline', '1px dashed rgba(120, 120, 120, 0.8)', 'important');
  }
  function clearEl(el) {
    el.removeAttribute('title'); delete el.dataset.aidStage;
    el.style.removeProperty('background-color'); el.style.removeProperty('outline');
  }

  function request(mode, items) {
    return new Promise((resolve) => chrome.runtime.sendMessage(
      { type: 'analyze', mode, minWords: settings.minWords, items,
        config: { backend: settings.backend, threads: settings.threads } },
      (res) => {
        if (chrome.runtime.lastError || !res || res.error) {
          console.warn('[AI detector]', chrome.runtime.lastError?.message || res?.error); resolve(null); return;
        }
        resolve(res);
      }));
  }
  function apply(res, final) {
    if (!res) return;
    for (const r of res.results) {
      if (final) cache.set(r.id, { ...r, demo: res.demo });
      const el = document.querySelector(`[${ATTR}="${r.id}"]`);
      if (el && active) paint(el, r, res.demo);
    }
  }
  // Сначала все новые видимые блоки получают быструю оценку, затем по 4 блока уточняются гибридным детектором
  let fullQueue = [];
  let fullRunning = false;
  async function flush() {
    timer = null;
    if (!queue.size) return;
    const batch = [...queue.entries()].slice(0, 50);
    batch.forEach(([id]) => queue.delete(id));
    const items = batch.map(([id, el]) => ({ id, text: el.innerText }));
    if (settings.twoStage) apply(await request('fast', items), false);
    fullQueue.push(...items);
    runFull();
    if (queue.size) schedule();
  }
  async function runFull() {
    if (fullRunning) return;
    fullRunning = true;
    while (fullQueue.length && active) {
      const items = fullQueue.splice(0, 4);
      apply(await request('full', items), true);
    }
    fullRunning = false;
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
    observer?.disconnect(); observer = null; queue.clear(); fullQueue = [];
    document.querySelectorAll(`[${ATTR}]`).forEach(clearEl);
  }

  // Оценка всей статьи: текст всех найденных блоков страницы объединяется и анализируется как один текст (512 токенов)
  async function analyzeArticle(s) {
    settings = { ...DEFAULTS, ...s };
    const text = collectBlocks().map((el) => el.innerText.trim()).join('\n');
    showBadge('Оцениваю статью целиком…');
    const res = await request('article', [{ id: 'article', text }]);
    const r = res?.results?.[0];
    if (!r || r.status !== 'ok') { showBadge(r?.status === 'unsupported_language' ? 'Язык страницы не поддерживается' : 'Не удалось оценить статью'); return; }
    const level = r.thr ? (r.p >= r.thr.fpr02 ? 'вероятно ИИ' : r.p >= r.thr.fpr10 ? 'возможно ИИ' : 'скорее человек') : '';
    showBadge(`Статья целиком: ${Math.round(r.p * 100)}% (${level})\nПроанализировано токенов: ${r.tokens}\n` +
      r.top.map((t) => `• ${t.label} — ${t.contribution > 0 ? 'похоже на ИИ' : 'похоже на человека'}`).join('\n') +
      '\nОценка вероятностная и не является доказательством.');
  }
  function showBadge(text) {
    let b = document.getElementById('aid-badge');
    if (!b) {
      b = document.createElement('div'); b.id = 'aid-badge';
      b.style.cssText = 'position:fixed;top:12px;right:12px;z-index:2147483647;max-width:340px;padding:10px 12px;' +
        'background:#111827;color:#f9fafb;font:13px/1.4 system-ui,sans-serif;border-radius:8px;white-space:pre-line;' +
        'box-shadow:0 4px 16px rgba(0,0,0,.3);cursor:pointer';
      b.title = 'Нажмите, чтобы закрыть';
      b.onclick = () => b.remove();
      document.documentElement.appendChild(b);
    }
    b.textContent = text;
  }

  chrome.runtime.onMessage.addListener((msg, _s, sendResponse) => {
    if (msg?.type === 'article') analyzeArticle(msg.settings);
    if (msg?.type === 'run') { settings = { ...DEFAULTS, ...msg.settings }; stop(); cache.clear(); start(); }
    if (msg?.type === 'clear') { stop(); document.getElementById('aid-badge')?.remove(); }
    if (msg?.type === 'status') sendResponse({ active });
  });
})();
