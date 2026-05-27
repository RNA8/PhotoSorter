let current = null;
let keepIds = new Set();

async function loadNext() {
  const r = await fetch('/api/moments/next');
  const data = await r.json();
  if (data.done) {
    document.getElementById('grid').innerHTML =
      '<div id="done-screen">All moments reviewed! Curated photos are in output/curated/</div>';
    document.getElementById('footer').style.display = 'none';
    document.getElementById('progress-label').textContent = 'Complete';
    return;
  }
  current = data;
  keepIds = new Set(data.photos.filter(p => p.suggested_keep).map(p => p.photo_id));
  setProgress(data.reviewed, data.total);
  renderGrid(data.photos);
}

function setProgress(reviewed, total) {
  const pct = total ? (reviewed / total * 100).toFixed(1) : 0;
  document.getElementById('progress-bar').style.width = pct + '%';
  document.getElementById('progress-label').textContent =
    `Moment ${reviewed + 1} of ${total} — ${pct}% complete`;
}

function renderGrid(photos) {
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  photos.forEach((p, i) => {
    const card = document.createElement('div');
    card.className = 'card' + (keepIds.has(p.photo_id) ? ' keep' : '');
    card.dataset.id = p.photo_id;
    card.innerHTML = `
      <div class="badge">${i + 1}</div>
      <img src="/api/photos/${p.photo_id}" loading="lazy" alt="${p.filename}">
      <div class="card-info">
        <div class="card-rank">#${p.rank} &mdash; ${pct(p.scores.composite)}%</div>
        ${bar('Gaze', p.scores.gaze)}
        ${bar('Smile', p.scores.smile)}
        ${bar('Eyes', p.scores.eyes)}
        ${bar('Sharpness', p.scores.sharpness)}
        ${bar('Exposure', p.scores.exposure)}
      </div>`;
    card.addEventListener('click', () => toggle(p.photo_id));
    grid.appendChild(card);
  });
}

function pct(v) { return ((v || 0) * 100).toFixed(0); }

function bar(label, value) {
  return `<div class="bar-row">
    <span class="bar-label">${label}</span>
    <div class="bar-bg"><div class="bar-fill" style="width:${pct(value)}%"></div></div>
    <span class="bar-val">${pct(value)}%</span>
  </div>`;
}

function toggle(id) {
  keepIds.has(id) ? keepIds.delete(id) : keepIds.add(id);
  document.querySelectorAll('.card').forEach(c => {
    c.className = 'card' + (keepIds.has(parseInt(c.dataset.id)) ? ' keep' : '');
  });
}

async function submit() {
  if (!current) return;
  await fetch(`/api/moments/${current.moment_id}/submit`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({keep_ids: [...keepIds]}),
  });
  loadNext();
}

async function undo() {
  await fetch('/api/undo', {method: 'POST'});
  loadNext();
}

document.getElementById('submit-btn').addEventListener('click', submit);

document.addEventListener('keydown', e => {
  if (e.key === 'ArrowRight' || e.key === 'Enter') { submit(); return; }
  if (e.key === 'z' || e.key === 'Z') { undo(); return; }
  if (e.key === 'a' || e.key === 'A') {
    const allKept = current?.photos.every(p => keepIds.has(p.photo_id));
    if (allKept) keepIds.clear(); else current?.photos.forEach(p => keepIds.add(p.photo_id));
    renderGrid(current?.photos || []);
    return;
  }
  const n = parseInt(e.key);
  if (n >= 1 && n <= 9 && current?.photos[n - 1]) toggle(current.photos[n - 1].photo_id);
});

loadNext();
