"""Single-page browser client served at `/`.

Kept as one self-contained string with no external assets so the API has no
static-files dependency and works offline.
"""

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Samsung Phone Query & Review System</title>
<style>
  :root {
    --bg: #0f1115; --panel: #171a21; --panel-2: #1e222b; --border: #2a2f3a;
    --text: #e6e8ec; --muted: #9aa3b2; --accent: #4c8dff; --accent-2: #23c48e;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  header {
    padding: 22px 28px; border-bottom: 1px solid var(--border);
    background: var(--panel); display: flex; align-items: baseline; gap: 14px;
    flex-wrap: wrap;
  }
  header h1 { margin: 0; font-size: 19px; letter-spacing: .2px; }
  header .sub { color: var(--muted); font-size: 13px; }
  header .status { margin-left: auto; font-size: 12px; color: var(--muted); }
  .dot { display:inline-block; width:8px; height:8px; border-radius:50%;
         background:#666; margin-right:6px; vertical-align:middle; }
  .dot.ok { background: var(--accent-2); }
  .dot.bad { background: #e5534b; }
  main { max-width: 1000px; margin: 0 auto; padding: 26px; }
  .tabs { display: flex; gap: 6px; margin-bottom: 18px; flex-wrap: wrap; }
  .tab {
    padding: 9px 16px; border: 1px solid var(--border); border-radius: 8px;
    background: var(--panel); color: var(--muted); cursor: pointer; font-size: 14px;
  }
  .tab:hover { color: var(--text); }
  .tab.active { background: var(--accent); border-color: var(--accent); color: #fff; }
  .panel { background: var(--panel); border: 1px solid var(--border);
           border-radius: 12px; padding: 20px; }
  label { display:block; font-size:12px; text-transform:uppercase;
          letter-spacing:.6px; color: var(--muted); margin: 12px 0 6px; }
  input, select, textarea {
    width: 100%; padding: 11px 13px; background: var(--panel-2);
    border: 1px solid var(--border); border-radius: 8px; color: var(--text);
    font: inherit;
  }
  input:focus, select:focus, textarea:focus { outline: none; border-color: var(--accent); }
  .row { display: flex; gap: 14px; flex-wrap: wrap; }
  .row > div { flex: 1 1 200px; }
  button.go {
    margin-top: 16px; padding: 11px 22px; background: var(--accent); color:#fff;
    border: none; border-radius: 8px; font: inherit; font-weight: 600; cursor: pointer;
  }
  button.go:hover { filter: brightness(1.1); }
  button.go:disabled { opacity: .55; cursor: progress; }
  .chips { display:flex; gap:8px; flex-wrap:wrap; margin-top:12px; }
  .chip {
    font-size: 12.5px; padding: 6px 11px; border-radius: 999px; cursor: pointer;
    background: var(--panel-2); border: 1px solid var(--border); color: var(--muted);
  }
  .chip:hover { color: var(--text); border-color: var(--accent); }
  .out { margin-top: 20px; display: none; }
  .out.show { display: block; }
  .answer {
    background: var(--panel-2); border: 1px solid var(--border);
    border-left: 3px solid var(--accent); border-radius: 8px;
    padding: 16px 18px; white-space: pre-wrap; word-wrap: break-word;
  }
  .answer.mono { font-family: ui-monospace, Consolas, monospace; font-size: 13px;
                 overflow-x: auto; white-space: pre; }
  .meta { margin-top: 12px; font-size: 12.5px; color: var(--muted); }
  .badge {
    display:inline-block; padding:3px 9px; border-radius:6px; margin-right:6px;
    background: rgba(76,141,255,.14); color:#9dc0ff; font-size:12px;
  }
  table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 14px; }
  th, td { text-align: left; padding: 9px 10px; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-size: 12px; text-transform: uppercase;
       letter-spacing: .5px; }
  tbody tr:hover { background: var(--panel-2); }
  .err { border-left-color: #e5534b; color: #ffb4b0; }
</style>
</head>
<body>
<header>
  <h1>Samsung Phone Query &amp; Review System</h1>
  <span class="sub">GSMArena data · RAG chatbot · multi-agent reviews</span>
  <span class="status" id="status"><span class="dot"></span>checking…</span>
</header>

<main>
  <div class="tabs">
    <div class="tab active" data-tab="chat">Chatbot</div>
    <div class="tab" data-tab="review">Agent Review</div>
    <div class="tab" data-tab="compare">Compare</div>
    <div class="tab" data-tab="browse">Browse Database</div>
  </div>

  <!-- Chat -->
  <section class="panel" id="tab-chat">
    <label for="q">Ask anything about Samsung phones</label>
    <input id="q" placeholder="What are the camera specs of the Samsung Galaxy S23?">
    <div class="chips">
      <span class="chip">What are the camera specs of the Samsung Galaxy S23?</span>
      <span class="chip">Which Samsung phone has the best battery life?</span>
      <span class="chip">How does the Galaxy S23 compare to the S22 in terms of performance?</span>
      <span class="chip">What is the screen size of the Galaxy S22?</span>
      <span class="chip">Which phone is the cheapest?</span>
    </div>
    <button class="go" id="askBtn">Ask</button>
    <div class="out" id="chatOut">
      <div class="answer" id="chatAnswer"></div>
      <div class="meta" id="chatMeta"></div>
    </div>
  </section>

  <!-- Review -->
  <section class="panel" id="tab-review" hidden>
    <div class="row">
      <div>
        <label for="rPhone">Phone</label>
        <input id="rPhone" placeholder="Galaxy S23 Ultra" value="Galaxy S23 Ultra">
      </div>
      <div>
        <label for="rAud">Written for</label>
        <input id="rAud" placeholder="a mobile photographer" value="a general buyer">
      </div>
    </div>
    <button class="go" id="revBtn">Generate review</button>
    <div class="out" id="revOut">
      <div class="answer" id="revAnswer"></div>
      <div class="meta" id="revMeta"></div>
    </div>
  </section>

  <!-- Compare -->
  <section class="panel" id="tab-compare" hidden>
    <div class="row">
      <div><label for="cA">Phone A</label><input id="cA" value="Galaxy S23"></div>
      <div><label for="cB">Phone B</label><input id="cB" value="Galaxy S22"></div>
      <div>
        <label for="cF">Focus</label>
        <select id="cF">
          <option>overall</option><option selected>performance</option>
          <option>camera</option><option>battery</option>
          <option>display</option><option>price</option>
        </select>
      </div>
    </div>
    <button class="go" id="cmpBtn">Compare</button>
    <div class="out" id="cmpOut">
      <div class="answer mono" id="cmpTable"></div>
      <div class="answer" id="cmpAnswer" style="margin-top:14px"></div>
      <div class="meta" id="cmpMeta"></div>
    </div>
  </section>

  <!-- Browse -->
  <section class="panel" id="tab-browse" hidden>
    <button class="go" id="loadBtn" style="margin-top:0">Load phones</button>
    <div class="out" id="brOut"><div id="brTable"></div></div>
  </section>
</main>

<script>
const $ = (id) => document.getElementById(id);

// Tabs
document.querySelectorAll('.tab').forEach(tab => {
  tab.onclick = () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    ['chat','review','compare','browse'].forEach(name => {
      $('tab-' + name).hidden = (name !== tab.dataset.tab);
    });
  };
});

// Sample-question chips
document.querySelectorAll('.chip').forEach(chip => {
  chip.onclick = () => { $('q').value = chip.textContent; ask(); };
});

async function call(path, body) {
  const options = body
    ? { method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body) }
    : {};
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({detail: 'Invalid response'}));
  if (!response.ok) throw new Error(data.detail || ('HTTP ' + response.status));
  return data;
}

function busy(button, on, idleLabel) {
  button.disabled = on;
  button.textContent = on ? 'Working…' : idleLabel;
}

function showError(box, target, message) {
  box.classList.add('show');
  target.classList.add('err');
  target.textContent = 'Error: ' + message;
}

// Chat
async function ask() {
  const query = $('q').value.trim();
  if (!query) return;
  busy($('askBtn'), true, 'Ask');
  $('chatOut').classList.add('show');
  $('chatAnswer').classList.remove('err');
  $('chatAnswer').textContent = 'Thinking…';
  $('chatMeta').textContent = '';
  try {
    const data = await call('/chat', {query});
    $('chatAnswer').textContent = data.answer;
    const sources = (data.sources || [])
      .map(s => s.phone + ' / ' + s.aspect).slice(0, 6);
    $('chatMeta').innerHTML =
      '<span class="badge">intent: ' + data.intent + '</span>' +
      '<span class="badge">via: ' + data.generated_by + '</span>' +
      (sources.length ? '<br>Sources: ' + sources.join(' · ') : '');
  } catch (e) { showError($('chatOut'), $('chatAnswer'), e.message); }
  busy($('askBtn'), false, 'Ask');
}
$('askBtn').onclick = ask;
$('q').addEventListener('keydown', e => { if (e.key === 'Enter') ask(); });

// Review
$('revBtn').onclick = async () => {
  busy($('revBtn'), true, 'Generate review');
  $('revOut').classList.add('show');
  $('revAnswer').classList.remove('err');
  $('revAnswer').textContent = 'The agents are working…';
  $('revMeta').textContent = '';
  try {
    const data = await call('/agents/review',
      {phone: $('rPhone').value, audience: $('rAud').value});
    $('revAnswer').textContent = data.review;
    const agents = Object.values(data.agents || {})
      .map(a => a.agent + ' (' + a.duration_seconds + 's)');
    $('revMeta').innerHTML =
      '<span class="badge">' + data.phone + '</span>' +
      '<span class="badge">via: ' + data.generated_by + '</span>' +
      '<br>Agents: ' + agents.join(' → ');
  } catch (e) { showError($('revOut'), $('revAnswer'), e.message); }
  busy($('revBtn'), false, 'Generate review');
};

// Compare
$('cmpBtn').onclick = async () => {
  busy($('cmpBtn'), true, 'Compare');
  $('cmpOut').classList.add('show');
  $('cmpAnswer').classList.remove('err');
  $('cmpTable').textContent = '';
  $('cmpAnswer').textContent = 'Comparing…';
  $('cmpMeta').textContent = '';
  try {
    const data = await call('/agents/compare',
      {phone_a: $('cA').value, phone_b: $('cB').value, focus: $('cF').value});
    $('cmpTable').textContent = data.table || '';
    $('cmpAnswer').textContent = data.comparison;
    $('cmpMeta').innerHTML =
      '<span class="badge">focus: ' + data.focus + '</span>' +
      '<span class="badge">via: ' + data.generated_by + '</span>';
  } catch (e) { showError($('cmpOut'), $('cmpAnswer'), e.message); }
  busy($('cmpBtn'), false, 'Compare');
};

// Browse
$('loadBtn').onclick = async () => {
  busy($('loadBtn'), true, 'Load phones');
  $('brOut').classList.add('show');
  try {
    const phones = await call('/phones');
    $('brTable').innerHTML =
      '<table><thead><tr><th>Model</th><th>Year</th><th>Display</th>' +
      '<th>Chipset</th><th>Camera</th><th>Battery</th></tr></thead><tbody>' +
      phones.map(p => '<tr>' +
        '<td>' + p.name + '</td>' +
        '<td>' + (p.release_year ?? '—') + '</td>' +
        '<td>' + (p.display_size_inches ?? '—') + '"</td>' +
        '<td>' + (p.chipset ?? '—') + '</td>' +
        '<td>' + (p.main_camera_mp ?? '—') + ' MP</td>' +
        '<td>' + (p.battery_capacity_mah ?? '—') + ' mAh</td>' +
      '</tr>').join('') + '</tbody></table>';
  } catch (e) { $('brTable').textContent = 'Error: ' + e.message; }
  busy($('loadBtn'), false, 'Load phones');
};

// Header status
(async () => {
  try {
    const health = await call('/health');
    const ok = health.status === 'healthy';
    $('status').innerHTML =
      '<span class="dot ' + (ok ? 'ok' : 'bad') + '"></span>' +
      health.database.phones + ' phones · ' +
      health.vector_store.documents + ' passages · ' +
      (health.llm.enabled ? health.llm.model.split('/').pop() : 'templates');
  } catch (e) {
    $('status').innerHTML = '<span class="dot bad"></span>API unreachable';
  }
})();
</script>
</body>
</html>
"""
