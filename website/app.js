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
  viewerRows: [],
  selected: null,
  request: 0,
  trace: null,
};
let data, methods, groups;
const pct = (v) => (Number.isFinite(v) ? v.toFixed(2) : '—');
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
function order(list, key, desc, get) {
  return [...list].sort((a, b) => {
    const x = get(a, key),
      y = get(b, key);
    if (x == null || y == null) return x == null ? (y == null ? 0 : 1) : -1;
    return (typeof x === 'string' ? x.localeCompare(y) : x - y) * (desc ? -1 : 1);
  });
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
  const search = $('search').value.trim().toLowerCase().replaceAll('_', ' '),
    v = $('variant').value;
  const filtered = groups
    .filter((g) => (!state.dimension || g.dimension === state.dimension) && (!v || v === g.variant))
    .filter(
      (g) =>
        g.rows.length &&
        (!search ||
          [g.name, g.id].some((x) => x.toLowerCase().replaceAll('_', ' ').includes(search))),
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
    const b = button('', () => openGroup(g.id), 'task');
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
    meta.append(make('span', 'variant', g.variant), make('span', '', g.dimension));
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
  methods = data.methods;
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
  $('stats-note').textContent =
    'PRIORA pools the final outcome for each evaluated layout, including designated reruns and adjudicated outcomes. Each dimension averages its task-family rates; the five-dimension mean weights dimensions equally. The gallery is a selection of examples.';
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
      td.title =
        m.name +
        ' · ' +
        r.name +
        (r.variant ? ' · ' + r.variant : '') +
        ' · ' +
        (v == null ? 'Not reported' : pct(v) + '%');
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
    'Final outcome per evaluated layout. Results as of ' + data.date + '.',
    'https://github.com/EchO-NN/PRIORA',
  );
  for (const m of data.methods.filter((m) => m.id !== 'priora')) {
    entry(m.name, m.protocol, m.url);
    for (const source of m.sources.filter((s) => s.url !== m.url)) {
      const dd = list.lastElementChild.querySelector('dd');
      dd.append(' · ', safeLink(source.label, source.url));
    }
  }
  $('source-list').replaceChildren(list);
}
function openGroup(key) {
  const g = groups.find((g) => g.id === key);
  if (!g) return;
  state.viewerRows = g.rows;
  if (!state.viewerRows.length) return;
  $('viewer-title').textContent = g.name + ' · ' + g.variant;
  $('viewer-dim').textContent = g.dimension;
  if (!$('viewer').open) $('viewer').showModal();
  chooseEpisode(state.viewerRows[0]);
}
function renderEpisodeList() {
  const list = $('episode-list');
  list.replaceChildren();
  for (const ep of state.viewerRows) {
    const b = button('', () => chooseEpisode(ep), 'episode', state.selected?.id === ep.id);
    const img = make('img');
    img.loading = 'lazy';
    img.alt = '';
    img.src = ep.poster;
    b.append(img, make('b', '', 'Example ' + ep.number));
    list.append(b);
  }
}
const cameraNames = { main: 'Main', left: 'Left wrist', right: 'Right wrist', all: 'Three views' };
let stopWristRefresh;
function refreshWristViews() {
  stopWristRefresh?.();
  stopWristRefresh = undefined;
  const p = $('player');
  if ($('wrist-pair').hidden || !state.selected || p.readyState < 2) return;
  const width = p.videoWidth / 3;
  const canvases = $('wrist-pair').querySelectorAll('canvas');
  canvases.forEach((canvas, index) => {
    canvas
      .getContext('2d')
      .drawImage(
        p,
        (index + 1) * width,
        0,
        width,
        p.videoHeight,
        0,
        0,
        canvas.width,
        canvas.height,
      );
  });
  if (p.paused || p.ended) return;
  if (p.requestVideoFrameCallback) {
    const id = p.requestVideoFrameCallback(refreshWristViews);
    stopWristRefresh = () => p.cancelVideoFrameCallback(id);
  } else {
    const id = requestAnimationFrame(refreshWristViews);
    stopWristRefresh = () => cancelAnimationFrame(id);
  }
}
function selectCamera(camera) {
  const stage = $('video-stage');
  stage.dataset.camera = camera;
  $('wrist-pair').hidden = camera !== 'all';
  refreshWristViews();
  $('player').setAttribute('aria-label', 'Recorded robot simulation, ' + cameraNames[camera]);
  for (const b of document.querySelectorAll('.camera-controls button'))
    b.setAttribute('aria-pressed', String(b.dataset.camera === camera));
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
    $('current-mode').textContent = t.excerpt ? 'Clip start' : 'Initial observation';
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
  selectCamera('main');
  const u = $('unavailable');
  p.pause();
  p.removeAttribute('src');
  p.removeAttribute('poster');
  p.load();
  const available = !!ep.video;
  p.hidden = !available;
  u.hidden = available;
  u.textContent = available ? '' : 'Recording unavailable for this example.';
  if (available) {
    p.src = ep.video;
    p.load();
  }
  const meta = $('episode-meta');
  meta.replaceChildren(make('b', '', 'Example ' + ep.number));
  renderEpisodeList();
  drawTrace(null);
  if (!ep.trace) return;
  try {
    const res = await fetch(ep.trace);
    if (!res.ok) throw Error(res.status);
    const t = await res.json();
    if (state.request === key) {
      drawTrace(t);
      if (t.excerpt || t.partial)
        meta.append(make('span', '', t.partial ? 'Partial recording' : 'Rollout excerpt'));
    }
  } catch {
    if (state.request === key) drawTrace(null);
  }
}
function initViewer() {
  const d = $('viewer'),
    p = $('player'),
    tl = $('timeline');
  for (const b of document.querySelectorAll('.camera-controls button'))
    b.onclick = () => selectCamera(b.dataset.camera);
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
    refreshWristViews();
    p.pause();
    p.removeAttribute('src');
    p.load();
  });
  for (const name of ['timeupdate', 'loadedmetadata', 'seeking', 'seeked'])
    p.addEventListener(name, updateTrace);
  for (const name of ['play', 'pause', 'loadeddata', 'seeked'])
    p.addEventListener(name, refreshWristViews);
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
  const r = await fetch('data/catalog.json?v=examples-2');
  if (!r.ok) throw Error(r.status);
  data = await r.json();
  groups = data.families.flatMap((f) =>
    f.variants.map((v) => ({
      id: f.id + '__' + v,
      task: f.id,
      name: f.name,
      dimension: f.dimension,
      variant: v,
      rows: Array.from({ length: 6 }, (_, i) => {
        const number = i + 1,
          id = f.id + '--' + v + '--' + number;
        return {
          id,
          number,
          video: 'media/examples/' + id + '.mp4',
          poster: 'media/posters/' + id + '.jpg',
          trace: 'data/examples/' + id + '.json',
        };
      }),
    })),
  );
  initGallery();
  initViewer();
  for (const d of ['', ...dimensions])
    $('heat-dimension').add(new Option(d || 'All dimensions', d));
  $('heat-dimension').onchange = renderMatrix;
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
