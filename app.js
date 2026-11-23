'use strict';
const $ = (id) => document.getElementById(id);
const make = (tag, className, text) => {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
};
const dimensions = ['Generalization', 'Precision', 'Long-Horizon', 'Memory', 'Open'];
const modes = { wall: 'WALL', edit: 'GPT edit', eef: 'GPT takeover', unknown: 'Unrecorded' };
const state = {
  dimension: '',
  page: 0,
  statsSort: 'avg',
  statsDesc: true,
  heatSort: 'task',
  heatDesc: false,
  level: 'family',
  batch: 'all',
  viewerRows: [],
  selected: null,
  sheet: 0,
  request: 0,
  trace: null,
};
let data, methods, groups;
const pct = (v) => (Number.isFinite(v) ? v.toFixed(2) : '—');
const fmtTask = (s) => s.replaceAll('_', ' ').replace(/^./, (x) => x.toUpperCase());
const takeMean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);
const selectedPanel = (e, p) => p === 'all' || e.panel === p;
function button(label, handler, cls, pressed) {
  const b = make('button', cls, label);
  b.type = 'button';
  b.onclick = handler;
  if (pressed !== undefined) b.setAttribute('aria-pressed', String(pressed));
  return b;
}
function safeLink(label, url) {
  const a = make('a', '', label);
  a.href = url;
  return a;
}
function resultKey(e) {
  return e.task + '__' + e.variant;
}
function countRate(rows) {
  const complete = rows.filter((e) => typeof e.success === 'boolean' && !e.displayOnly);
  return {
    n: complete.length,
    k: complete.filter((e) => e.success).length,
    rate: complete.length
      ? (100 * complete.filter((e) => e.success).length) / complete.length
      : null,
  };
}
function order(list, key, desc, get) {
  return [...list].sort((a, b) => {
    const x = get(a, key),
      y = get(b, key);
    if (x == null || y == null) return x == null ? (y == null ? 0 : 1) : -1;
    return (typeof x === 'string' ? x.localeCompare(y) : x - y) * (desc ? -1 : 1);
  });
}
function prioraMetrics() {
  const rows = data.episodes.filter((e) => selectedPanel(e, state.batch) && !e.displayOnly);
  const family = {},
    variant = {},
    dims = {},
    counts = {},
    variantCounts = {};
  for (const f of data.families) {
    const matched = rows.filter((e) => e.task === f.id);
    const r = countRate(matched);
    family[f.id] = r.rate;
    counts[f.id] = r;
    for (const v of f.variants) {
      const t = countRate(matched.filter((e) => e.variant === v));
      variant[f.id + '__' + v] = t.rate;
      variantCounts[f.id + '__' + v] = t;
    }
  }
  for (const d of dimensions)
    dims[d] = takeMean(
      data.families
        .filter((f) => f.dimension === d)
        .map((f) => family[f.id])
        .filter(Number.isFinite),
    );
  return {
    id: 'priora',
    name: 'PRIORA',
    family,
    variant,
    dims,
    avg: dimensions.every((d) => dims[d] != null) ? takeMean(Object.values(dims)) : null,
    counts,
    variantCounts,
  };
}
function pager(el, page, n, change, size, unit) {
  el.replaceChildren();
  const pages = Math.ceil(n / size);
  if (pages <= 1) return;
  const prev = button('Previous', () => change(page - 1));
  prev.disabled = page === 0;
  const next = button('Next', () => change(page + 1));
  next.disabled = page >= pages - 1;
  el.append(prev, make('span', '', unit + ' ' + (page + 1) + ' of ' + pages), next);
}
function setGalleryPage(page) {
  state.page = page;
  renderGallery();
}
function renderGallery() {
  const panel = $('panel').value,
    search = $('search').value.trim().toLowerCase().replaceAll('_', ' '),
    v = $('variant').value;
  const filtered = groups
    .filter((g) => (!state.dimension || g.dimension === state.dimension) && (!v || v === g.variant))
    .map((g) => ({ ...g, rows: g.rows.filter((e) => selectedPanel(e, panel)) }))
    .filter(
      (g) =>
        g.rows.length &&
        (!search ||
          [g.name, g.id, ...g.rows.map((e) => e.id)].some((x) =>
            x.toLowerCase().replaceAll('_', ' ').includes(search),
          )),
    );
  const grid = $('gallery-grid');
  grid.replaceChildren();
  $('gallery-count').textContent = filtered.length + ' task variants';
  if (!filtered.length) {
    grid.append(make('p', 'empty', 'No task matches these filters.'));
    $('gallery-pager').replaceChildren();
    return;
  }
  const size = 12;
  state.page = Math.min(state.page, Math.max(0, Math.ceil(filtered.length / size) - 1));
  for (const g of filtered.slice(state.page * size, (state.page + 1) * size)) {
    const b = button('', () => openGroup(g.id, panel), 'task');
    const poster = g.rows.find((e) => e.poster);
    if (poster) {
      const pic = make('img', 'photo');
      pic.alt = '';
      pic.loading = 'lazy';
      pic.src = poster.poster;
      b.append(pic);
    } else b.append(make('span', 'photo blankphoto', 'Preview unavailable'));
    const block = make('span', 'task-body');
    block.append(make('strong', 'task-name', g.name));
    const meta = make('span', 'task-details');
    const clips = g.rows.filter((e) => e.video).length;
    meta.append(
      make('span', 'variant', g.variant),
      make('span', '', clips + ' recording' + (clips === 1 ? '' : 's')),
    );
    block.append(meta);
    b.append(block);
    grid.append(b);
  }
  pager($('gallery-pager'), state.page, filtered.length, setGalleryPage, size, 'Page');
}
function initGallery() {
  const options = ['', ...dimensions];
  const c = $('dimension-filters');
  for (const d of options)
    c.append(
      button(
        d || 'All dimensions',
        () => {
          state.dimension = d;
          state.page = 0;
          [...c.children].forEach((b, i) =>
            b.setAttribute('aria-pressed', String(options[i] === d)),
          );
          renderGallery();
        },
        '',
        !d,
      ),
    );
  $('search').oninput = () => setGalleryPage(0);
  $('variant').onchange = () => setGalleryPage(0);
  $('panel').onchange = () => setGalleryPage(0);
}
function header(thead, cols, active, desc, change) {
  const row = make('tr');
  for (const [key, label] of cols) {
    const th = make('th');
    th.scope = 'col';
    if (active === key) th.setAttribute('aria-sort', desc ? 'descending' : 'ascending');
    const b = button(label, () => change(key));
    b.append(make('span', 'arrow', active === key ? (desc ? '▾' : '▴') : '↕'));
    th.append(b);
    row.append(th);
  }
  thead.append(row);
}
function renderStats() {
  const ours = prioraMetrics();
  methods = [ours, ...data.methods];
  const t = $('summary');
  t.replaceChildren();
  const head = t.createTHead();
  const cols = [['name', 'Method'], ['avg', 'Five-dim. mean'], ...dimensions.map((d) => [d, d])],
    get = (m, k) => (k === 'name' ? m.name : k === 'avg' ? m.avg : m.dims[k]);
  header(head, cols, state.statsSort, state.statsDesc, (k) => {
    state.statsDesc = k === state.statsSort ? !state.statsDesc : k !== 'name';
    state.statsSort = k;
    renderStats();
  });
  const body = t.createTBody(),
    best = {};
  for (const [k] of cols.slice(1))
    best[k] = Math.max(...methods.map((m) => get(m, k)).filter(Number.isFinite));
  for (const m of order(methods, state.statsSort, state.statsDesc, get)) {
    const row = body.insertRow();
    if (m.id === 'priora') row.className = 'ours';
    const th = make('th', '', m.name);
    th.scope = 'row';
    row.append(th);
    for (const [k] of cols.slice(1)) {
      const v = get(m, k);
      row.append(make('td', v != null && Math.abs(v - best[k]) < 0.005 ? 'best' : '', pct(v)));
    }
  }
  const selected = data.episodes.filter((e) => selectedPanel(e, state.batch) && !e.displayOnly);
  $('eval-count').textContent =
    countRate(selected).n + ' completed / ' + selected.length + ' scheduled PRIORA layouts';
  $('stats-note').textContent =
    'PRIORA uses the final result per layout, including designated reruns and human adjudication. A dimension averages its task-family rates; the mean gives each of the five dimensions equal weight. Pending episodes are excluded. External methods retain their published evaluation panels.';
  renderMatrix();
}
const heatStops = [
  [244, 247, 252],
  [204, 222, 244],
  [145, 183, 230],
  [80, 126, 196],
  [36, 61, 135],
];
function tint(v) {
  const k = Math.min(4, Math.max(0, v / 25)),
    lo = Math.floor(k),
    hi = Math.min(4, lo + 1);
  return heatStops[lo].map((c, i) => Math.round(c + (heatStops[hi][i] - c) * (k - lo)));
}
function renderMatrix() {
  const table = $('matrix');
  table.replaceChildren();
  const head = table.createTHead();
  const cols = [['task', 'Task'], ...methods.map((m) => [m.id, m.name])];
  header(head, cols, state.heatSort, state.heatDesc, (k) => {
    state.heatDesc = k === state.heatSort ? !state.heatDesc : k !== 'task';
    state.heatSort = k;
    renderMatrix();
  });
  const level = state.level,
    which = $('heat-dimension').value;
  const list =
    level === 'family'
      ? data.families.map((f) => ({ key: f.id, id: f.id, name: f.name, dimension: f.dimension }))
      : groups.map((g) => ({ ...g, key: g.id }));
  const lookup = (r, k) =>
    k === 'task'
      ? r.name + (r.variant || '')
      : (methods.find((m) => m.id === k)?.[level]?.[r.key] ?? null);
  const body = table.createTBody();
  for (const r of order(
    list.filter((r) => !which || r.dimension === which),
    state.heatSort,
    state.heatDesc,
    lookup,
  )) {
    const row = body.insertRow(),
      th = make('th');
    th.scope = 'row';
    const link = button(
      r.name,
      () =>
        openGroup(
          level === 'variant'
            ? r.key
            : (
                groups.find((g) => g.task === r.id && g.variant === 'standard') ||
                groups.find((g) => g.task === r.id)
              ).id,
          state.batch,
        ),
      'row-link',
    );
    th.append(link);
    if (r.variant) th.append(make('span', 'variant', r.variant));
    row.append(th);
    for (const m of methods) {
      const v = lookup(r, m.id),
        td = make(
          'td',
          v == null ? 'missing' : '',
          v == null ? '—' : Number(v.toFixed(1)).toString(),
        );
      if (v != null) {
        const c = tint(v);
        td.style.backgroundColor = 'rgb(' + c.join(',') + ')';
        td.style.color = c[1] < 145 ? 'white' : '#172744';
      }
      const detail = (level === 'family' ? m.counts : m.variantCounts)?.[r.key];
      td.title =
        m.name +
        ' · ' +
        r.name +
        (r.variant ? ' · ' + r.variant : '') +
        ' · ' +
        (v == null ? 'Not reported' : pct(v) + '%') +
        (detail?.n ? ' (' + detail.k + '/' + detail.n + ')' : '');
      row.append(td);
    }
  }
}
function renderSources() {
  const list = make('dl');
  function entry(name, desc, url) {
    const row = make('div'),
      dt = make('dt');
    dt.append(url ? safeLink(name, url) : document.createTextNode(name));
    row.append(dt, make('dd', '', desc));
    list.append(row);
  }
  entry(
    'PRIORA',
    'Final outcome per unique layout from the original 252 and additional 210 panels. Snapshot ' +
      data.date +
      '.',
    'https://github.com/EchO-NN/PRIORA/tree/main/priora/benchmarks',
  );
  for (const m of data.methods) {
    entry(m.name, m.protocol, m.url);
    for (const source of m.sources.filter((s) => s.url !== m.url)) {
      const dd = list.lastElementChild.querySelector('dd');
      dd.append(' · ', safeLink(source.label, source.url));
    }
  }
  $('source-list').replaceChildren(list);
}
function displayRows(g, panel) {
  const rows = g.rows.filter((e) => selectedPanel(e, panel));
  // Alternate panels where available so the first six cover individual layouts from both batches.
  const out = [];
  for (const category of [
    (e) => e.video && !e.displayOnly && e.success != null,
    (e) => e.video && !e.displayOnly && e.success == null,
    (e) => e.video && e.displayOnly,
    (e) => !e.video,
  ]) {
    const bins = ['252', '210'].map((p) => rows.filter((e) => e.panel === p && category(e)));
    for (let i = 0; i < Math.max(...bins.map((a) => a.length)); i++)
      for (const a of bins) if (a[i]) out.push(a[i]);
  }
  return out;
}
function openGroup(key, panel) {
  const g = groups.find((g) => g.id === key);
  if (!g) return;
  state.viewerRows = displayRows(g, panel);
  if (!state.viewerRows.length) return;
  state.sheet = 0;
  $('viewer-title').textContent = g.name + ' · ' + g.variant;
  $('viewer-dim').textContent = g.dimension;
  if (!$('viewer').open) $('viewer').showModal();
  chooseEpisode(state.viewerRows[0]);
}
function renderEpisodeList() {
  const el = $('episode-list');
  el.replaceChildren();
  const rows = state.viewerRows;
  const n = rows.length;
  $('episode-total').textContent = n + ' attempt' + (n === 1 ? '' : 's');
  for (const ep of rows.slice(state.sheet * 6, state.sheet * 6 + 6)) {
    const b = button('', () => chooseEpisode(ep), 'episode', state.selected?.id === ep.id);
    let img;
    if (ep.poster) {
      img = make('img');
      img.loading = 'lazy';
      img.src = ep.poster;
      img.alt = '';
    } else img = make('span', 'mini-placeholder');
    const body = make('span');
    body.append(
      make('b', '', 'Layout ' + ep.layout + ' · group ' + ep.group),
      make(
        'small',
        '',
        ep.panel +
          ' panel · ' +
          (ep.displayOnly
            ? ep.extraLabel || 'Earlier attempt'
            : ep.success === true
              ? 'Success'
              : ep.success === false
                ? 'Failure'
                : ep.extraLabel || 'No terminal result') +
          (ep.video ? '' : ' · No video'),
      ),
    );
    b.append(img, body);
    el.append(b);
  }
  pager(
    $('episode-pager'),
    state.sheet,
    n,
    (p) => {
      state.sheet = p;
      renderEpisodeList();
    },
    6,
    'Page',
  );
}
function seekRatio(f) {
  const p = $('player');
  if (!Number.isFinite(p.duration) || p.duration <= 0) return;
  p.currentTime = Math.max(0, Math.min(p.duration - 0.001, f * p.duration));
  updateTrace();
}
function updateTrace() {
  const p = $('player'),
    t = state.trace;
  if (!state.selected) return;
  const ratio =
    Number.isFinite(p.duration) && p.duration > 0
      ? Math.max(0, Math.min(1, p.currentTime / p.duration))
      : 0;
  const observed = t?.steps
    ? Math.max(0, Math.min(t.steps, Math.floor(p.currentTime / t.controlDt + 1e-6)))
    : 0;
  const position = Math.max(0, observed - 1);
  const panel = $('timeline');
  panel.setAttribute('aria-valuenow', String(Math.round(ratio * 100)));
  $('cursor').style.left = ratio * 100 + '%';
  if (!t || !t.steps || !t.segments.length) {
    $('current-step').textContent = '';
    $('current-mode').textContent = 'Unrecorded';
    $('subtask').textContent = 'No matched action trace for this recording.';
    return;
  }
  $('current-step').textContent = 'Step ' + observed + ' / ' + t.steps;
  if (!observed) {
    $('current-mode').textContent = 'Initial observation';
    $('subtask').textContent = t.subtasks[t.segments[0][3]] || 'Subtask not recorded';
    return;
  }
  const seg = t.segments.find((s) => s[0] <= position && position < s[1]);
  $('current-mode').textContent = modes[seg?.[2]] || modes.unknown;
  $('subtask').textContent =
    (seg?.[3] != null ? t.subtasks[seg[3]] : null) || 'Subtask not recorded';
}
function drawTrace(t) {
  state.trace = t;
  const box = $('segments');
  box.replaceChildren();
  if (t?.steps) {
    let at = 0;
    function add(w, mode) {
      if (w <= 0) return;
      const i = make('i', mode);
      i.style.width = (100 * w) / t.steps + '%';
      box.append(i);
    }
    for (const s of t.segments) {
      add(s[0] - at, 'unknown');
      add(s[1] - s[0], s[2]);
      at = s[1];
    }
    add(t.steps - at, 'unknown');
  }
  updateTrace();
}
async function chooseEpisode(ep) {
  const key = ++state.request,
    p = $('player');
  state.selected = ep;
  const u = $('unavailable');
  p.pause();
  p.removeAttribute('src');
  p.removeAttribute('poster');
  p.load();
  if (ep.poster) p.poster = ep.poster;
  const available = !!ep.video;
  p.hidden = !available;
  u.hidden = available;
  u.textContent = available ? '' : 'Recording unavailable for this layout.';
  if (available) {
    p.src = ep.video;
    p.load();
  }
  const m = $('episode-meta'),
    status =
      ep.success === true ? 'Success' : ep.success === false ? 'Failure' : 'No terminal result';
  m.replaceChildren(
    make('b', '', 'Layout ' + ep.layout + ' · group ' + ep.group),
    make('span', '', 'Panel ' + ep.panel),
    make(
      'span',
      'status ' + (ep.success === true ? 's-ok' : ep.success === false ? 's-fail' : ''),
      status,
    ),
  );
  if (ep.extraLabel)
    m.append(make('span', '', ep.extraLabel + (ep.displayOnly ? ' · gallery only' : '')));
  if (ep.mediaNote) m.append(make('span', '', ep.mediaNote));
  renderEpisodeList();
  drawTrace(null);
  if (!ep.trace) return;
  try {
    const res = await fetch(ep.trace);
    if (!res.ok) throw Error(res.status);
    const t = await res.json();
    if (state.request === key) drawTrace(t);
  } catch {
    if (state.request === key) drawTrace(null);
  }
}
function initViewer() {
  const d = $('viewer'),
    p = $('player'),
    tl = $('timeline');
  $('close-viewer').onclick = () => d.close();
  d.onclick = (e) => {
    if (e.target === d) {
      const r = d.getBoundingClientRect();
      if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom)
        d.close();
    }
  };
  d.addEventListener('close', () => {
    ++state.request;
    state.selected = null;
    p.pause();
    p.removeAttribute('src');
    p.load();
  });
  for (const name of ['timeupdate', 'loadedmetadata', 'seeking', 'seeked'])
    p.addEventListener(name, updateTrace);
  p.addEventListener('error', () => {
    if (!state.selected?.video) return;
    $('unavailable').hidden = false;
    $('unavailable').textContent =
      'Video could not load. Check your connection or select another recording.';
  });
  const move = (e) => {
    const r = tl.getBoundingClientRect();
    seekRatio((e.clientX - r.left) / r.width);
  };
  tl.onpointerdown = (e) => {
    tl.setPointerCapture(e.pointerId);
    move(e);
  };
  tl.onpointermove = (e) => {
    if (tl.hasPointerCapture(e.pointerId)) move(e);
  };
  tl.onkeydown = (e) => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) return;
    e.preventDefault();
    if (Number.isFinite(p.duration))
      seekRatio(
        e.key === 'Home'
          ? 0
          : e.key === 'End'
            ? 1
            : (p.currentTime + (e.key === 'ArrowLeft' ? -1 : 1)) / p.duration,
      );
  };
}
async function main() {
  const r = await fetch('data/catalog.json');
  if (!r.ok) throw Error(r.status);
  data = await r.json();
  groups = data.families.flatMap((f) =>
    f.variants.map((v) => ({
      id: f.id + '__' + v,
      task: f.id,
      name: f.name,
      dimension: f.dimension,
      variant: v,
      rows: data.episodes.filter((e) => e.task === f.id && e.variant === v),
    })),
  );
  initGallery();
  initViewer();
  for (const d of ['', ...dimensions])
    $('heat-dimension').add(new Option(d || 'All dimensions', d));
  $('heat-dimension').onchange = renderMatrix;
  $('stats-panel').onchange = () => {
    state.batch = $('stats-panel').value;
    renderStats();
  };
  $('granularity').onclick = (e) => {
    const b = e.target.closest('button[data-level]');
    if (!b) return;
    state.level = b.dataset.level;
    for (const child of $('granularity').children)
      child.setAttribute('aria-pressed', String(child === b));
    renderMatrix();
  };
  renderGallery();
  renderStats();
  renderSources();
}
main().catch(() => {
  const e = $('load-error');
  e.hidden = false;
  e.textContent = 'The evaluation data could not be loaded. Please reload this page.';
});
