'use strict';
const $ = id => document.getElementById(id);
let rows = [], displayed = [], paused = false, filter = 'all', retry = 800, socket, runId = null, connected = false;
let clientFilter = null, clients = [], clientsAvailable = false, clientSignature = '';
const pendingMutes = new Set();
let clearingClients = false;
const number = new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 });
function category(e) {
  const n = e.event;
  if (/Error|EmptyResponse/.test(n) || e.data.status === 'failed' || e.data.status === 'rejected') return 'error';
  if (/Live.Request|SPK|ASR|Utterance/.test(n)) return 'speech';
  if (/Live.Tool|Live.Command|Control/.test(n)) return 'tools';
  if (/Live.Reply|Live.Audio|TTS/.test(n)) return 'output';
  return 'engine';
}
function describe(e) {
  const d = e.data, n = e.event;
  if (n === 'Live.Request') return [d.source === 'voice' ? 'Speech received' : 'Text request', d.text || '(empty input)'];
  if (n === 'Live.Reply') return [d.text ? 'Aria replied' : 'Turn ended without speech', d.text || (d.error || 'No spoken reply for this turn.')];
  if (n === 'Live.Model') return ['Model response', `${number.format((d.total_duration || 0)/1e9)} s · ${d.eval_count ?? '—'} generated tokens · ${(d.tool_calls || []).length} tool calls`];
  if (n === 'Live.Tool') return [`Tool · ${d.name}`, JSON.stringify(d.arguments || {})];
  if (n === 'Live.Command') return ['Command recorded', d.result?.dry_run ? `LOG ONLY · ${d.cmd_id} · no device changed` : `${d.cmd_id} · ${d.result?.outcome || 'see result'}`];
  if (n === 'Live.Audio') return ['Audio ready', 'Generated response audio is available.'];
  if (n === 'ARIA.Request.Status') return [`Request ${d.status}`, d.error || `Queue waiting: ${d.queue_pending ?? '—'}`];
  if (n === 'ARIA.LLM.Start') return ['Engine processing', `${(d.language || '').toUpperCase()} · ${d.source || 'request'} input`];
  if (n === 'ARIA.LLM.Done') return ['Engine finished', `${number.format((d.ms || 0)/1000)} s · ${d.tool_calls_made ?? 0} tool calls`];
  if (n === 'ARIA.SPK') return ['Speaker evidence', `${d.name || 'Unknown'} · score ${number.format(d.score || 0)} · ${d.known ? 'above threshold' : 'below threshold'}`];
  if (n === 'ARIA.Conversation') return ['Conversation state', `${d.action || ''} · ${d.state || d.reason || ''}`];
  if (n === 'ARIA.Control.Resolved') return ['Control resolved', d.status || 'See fields'];
  return [n.replace(/^ARIA\.|^Live\./, '').replaceAll('.', ' / '), d.error || d.text || d.reason || 'Expand to inspect the recorded fields.'];
}
function element(tag, cls, text) {
  const node = document.createElement(tag); if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text; return node;
}
function render() {
  const focused = document.activeElement?.closest('#events details')?.dataset.seq;
  const query = $('search').value.trim().toLowerCase();
  const selected = clientRows().filter(e => (filter === 'all' || category(e) === filter) && (!query || JSON.stringify(e.data).toLowerCase().includes(query)));
  const opened = new Set([...$('events').querySelectorAll('details[open]')].map(x => x.dataset.seq));
  const fragment = document.createDocumentFragment();
  for (const e of [...selected].reverse()) {
    const kind = category(e), [title, body] = describe(e);
    const card = element('details', `event ${kind}`); card.dataset.seq = String(e.seq); card.open = opened.has(String(e.seq));
    const summary = element('summary');
    const top = element('div', 'event-top'); top.append(element('span', 'event-badge', kind.toUpperCase()));
    if (e.client_id) top.append(element('span', 'client-tag', e.client_id));
    const time = element('time', '', new Date(e.time*1000).toLocaleTimeString([], {hour12:false})); time.dateTime = new Date(e.time*1000).toISOString(); top.append(time);
    summary.append(top, element('div', 'event-title', title), element('p', 'event-body', body));
    if (e.data.request_id) summary.append(element('div', 'event-id', `REQ ${e.data.request_id.slice(0,12)} · expand details`));
    card.append(summary, element('pre', '', JSON.stringify(e.data, null, 2))); fragment.append(card);
  }
  if (!selected.length) {
    const empty = element('div', 'empty'); empty.append(element('span', 'empty-mark', '⌁'), element('h3', '', displayed.length ? 'No matching events.' : 'Listening to the system.'), element('p', '', displayed.length ? 'Try another filter or search term.' : 'New transcripts, engine steps and replies will appear here when Aria is active.')); fragment.append(empty);
  }
  $('events').replaceChildren(fragment);
  if (focused) $('events').querySelector(`details[data-seq="${focused}"] summary`)?.focus({preventScroll:true});
  $('event-count').textContent = `${selected.length} EVENTS`;
  $('view-mode').textContent = paused ? 'VIEW PAUSED / FEED STILL RECEIVING' : 'NEWEST FIRST / UP TO 200 EVENTS';
  updateOverview();
}
function clientRows() { return clientFilter ? displayed.filter(e => e.client_id === clientFilter) : displayed; }
function latest(name) { return [...clientRows()].reverse().find(e => e.event === name); }
function updateOverview() {
  $('requests').textContent = String(clientRows().filter(e => e.event === 'Live.Request').length);
  const model = latest('Live.Model'); $('model-time').textContent = model ? `${number.format((model.data.total_duration || 0)/1e9)}s` : '—';
  const queue = latest('ARIA.Request.Status'); $('queue').textContent = queue?.data.queue_pending ?? '—';
  const input = latest('Live.Request'), reply = latest('Live.Reply');
  const recognition = input && displayed.find(e => e.event === 'ARIA.SPK.Request' && e.data.request_id === input.data.request_id);
  $('heard').textContent = input?.data.text || 'No transcript yet.';
  const person = input?.data.speaker_user || 'unknown';
  const recognized = recognition?.data.known && person !== 'unknown';
  const attributed = person !== 'unknown' && input?.data.source === 'voice';
  $('speaker').className = recognized ? 'speaker-known' : attributed ? 'speaker-nearest' : 'speaker-unknown';
  $('speaker').textContent = input ? (input.data.source !== 'voice' ? 'Text input' : `${recognized ? '' : attributed ? 'Nearest: ' : ''}${person}`) : '—';
  const matches = reply && input && reply.data.request_id === input.data.request_id;
  $('reply').textContent = matches ? (reply.data.text || 'No spoken reply for this turn.') : (input ? 'Awaiting this turn’s reply.' : 'No reply yet.');
  const score = recognition?.data.score;
  $('recognition').className = `recognition ${recognized ? 'speaker-known' : attributed ? 'speaker-nearest' : 'speaker-unknown'}`;
  $('recognition').textContent = typeof score === 'number' && Number.isFinite(score) ? `${number.format(score*100)}% similarity · ${recognition.data.known ? 'above recognition threshold' : 'below recognition threshold'} · not a probability` : (input?.data.source === 'voice' ? 'No recognition score for this turn.' : 'Speaker recognition applies to voice input.');
  const last = clientRows().at(-1); const kind = last ? category(last) : null;
  document.querySelectorAll('[data-stage]').forEach(n => n.classList.toggle('active', n.dataset.stage === kind));
  $('phase').textContent = last ? describe(last)[0] : 'Awaiting feed';
}
async function setClientMute(client) {
  pendingMutes.add(client.client_id); renderClients(true);
  $('client-control-message').textContent = '';
  try {
    const response = await fetch(`/api/clients/${encodeURIComponent(client.client_id)}/mute`, {
      method: 'POST', headers: {'Content-Type':'application/json', 'X-Aria-Control':'mute'},
      body: JSON.stringify({muted: !client.muted, run_id: runId}), signal: AbortSignal.timeout(4000)
    });
    if (!response.ok) throw new Error(`Server returned ${response.status}`);
    const state = await response.json();
    Object.assign(client, state);
    $('client-control-message').textContent = state.muted ? `${client.client_id}: incoming audio muted for 5 minutes.` : `${client.client_id}: incoming audio unmuted.`;
  } catch (error) {
    $('client-control-message').textContent = `Could not confirm mute change for ${client.client_id}. Check its live status. ${error.message}`;
  } finally { pendingMutes.delete(client.client_id); renderClients(true); }
}

async function clearOfflineClients() {
  clearingClients = true; renderClients(true);
  $('client-control-message').textContent = '';
  try {
    const response = await fetch('/api/clients/clear-offline', {
      method: 'POST', headers: {'Content-Type':'application/json', 'X-Aria-Control':'clear-offline'},
      body: JSON.stringify({run_id: runId}), signal: AbortSignal.timeout(4000)
    });
    if (!response.ok) throw new Error(`Server returned ${response.status}`);
    const state = await response.json();
    clients = state.clients;
    if (state.cleared.includes(clientFilter)) clientFilter = null;
    $('client-control-message').textContent = `${state.cleared.length} offline clients cleared. They reappear when they reconnect.`;
    render();
  } catch (error) {
    $('client-control-message').textContent = `Could not clear offline clients. Refresh status and retry. ${error.message}`;
  } finally { clearingClients = false; renderClients(true); }
}
function renderClients(force=false) {
  if (clientsAvailable && clientFilter && !clients.some(c => c.client_id === clientFilter)) {
    clientFilter = null; render();
  }
  $('clients').hidden = clients.length === 0;
  const offlineCount = clients.filter(c => !c.connected).length;
  const clear = $('clear-offline');
  clear.disabled = !clientsAvailable || clearingClients || offlineCount === 0;
  clear.textContent = clearingClients ? 'Clearing…' : `Clear offline (${offlineCount})`;
  clear.onclick = clearOfflineClients;
  const signature = JSON.stringify([clients, clientsAvailable, clientFilter]);
  if (!force && signature === clientSignature) return;
  clientSignature = signature;
  const focusedButton = document.activeElement?.closest('#clients button');
  const focused = focusedButton?.dataset.client, focusedAction = focusedButton?.dataset.action;
  const fragment = document.createDocumentFragment();
  const add = (name, status, tone, label) => {
    const button = element('button', `client-chip ${tone}`); button.type = 'button'; button.dataset.client = name || '';
    button.setAttribute('aria-pressed', String(clientFilter === name));
    button.append(element('i', 'client-dot'), element('span', 'client-name', label), element('small', '', status));
    button.onclick = () => { clientFilter = name; renderClients(true); render(); };
    fragment.append(button); return button;
  };
  add(null, String(clients.length), 'all-clients', 'All clients');
  for (const client of clients) {
    const remaining = Math.ceil(client.mute_remaining_s || 0);
    const countdown = `${Math.floor(remaining/60)}:${String(remaining%60).padStart(2,'0')}`;
    const state = !clientsAvailable ? 'Unknown' : client.muted ? `Muted · ${countdown}${client.connected ? '' : ' · offline'}` : !client.connected ? 'Offline' : client.transmitting ? 'Transmitting' : 'Connected';
    const activity = clientsAvailable && (client.processing ? ' · processing' : client.queued ? ' · queued' : '');
    const tone = !clientsAvailable ? 'unknown' : client.muted ? 'muted' : client.processing ? 'processing' : client.transmitting ? 'transmitting' : client.connected ? 'online' : 'offline';
    const chip = add(client.client_id, state + activity, tone, client.client_id);
    const group = element('div', 'client-control'); group.append(chip);
    const control = element('button', 'mute-button', pendingMutes.has(client.client_id) ? 'Saving…' : client.muted ? 'Unmute' : 'Mute 5 min');
    control.type = 'button'; control.dataset.client = client.client_id; control.dataset.action = 'mute';
    control.disabled = !clientsAvailable || pendingMutes.has(client.client_id);
    control.setAttribute('aria-label', `${client.muted ? 'Unmute' : 'Mute for 5 minutes'} ${client.client_id}`);
    control.title = 'Stops incoming audio processing. Already accepted requests finish normally.';
    control.onclick = () => setClientMute(client);
    group.append(control); fragment.append(group);
  }
  $('clients').replaceChildren(fragment);
  if (focused !== undefined) [...$('clients').querySelectorAll('button')].find(b => b.dataset.client === focused && b.dataset.action === focusedAction)?.focus({preventScroll:true});
  $('client-summary').textContent = clientsAvailable ? `${clients.filter(c=>c.connected).length} connected · ${clients.filter(c=>c.processing).length} processing` : 'Client status unavailable';
}
function connect() {
  socket = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws/events`);
  socket.onopen = () => { connected = true; retry = 800; $('connection').textContent = 'Feed connected'; $('connection').className = 'connected'; };
  socket.onmessage = message => {
    const packet = JSON.parse(message.data);
    if (runId !== packet.status.run_id) { clients = []; clientFilter = null; }
    if (packet.type === 'snapshot' || runId !== packet.status.run_id) { rows = []; if (!paused) displayed = []; }
    runId = packet.status.run_id;
    clientsAvailable = Boolean(packet.clients?.available);
    if (clientsAvailable) clients = packet.clients.clients;
    renderClients();
    const known = new Set(rows.map(e => e.seq));
    rows = [...rows, ...packet.events.filter(e => !known.has(e.seq))].sort((a,b) => a.seq-b.seq).slice(-200);
    $('run-id').textContent = runId || 'Waiting for Aria';
    $('source-state').textContent = packet.status.source === 'reading' ? 'Reading the Aria log source' : 'Log source unavailable';
    $('tool-mode').textContent = packet.status.tools_mode === 'log_only' ? 'Tools · log only' : `Tools · ${packet.status.tools_mode}`;
    if (!paused && (packet.events.length || packet.type === 'snapshot')) { displayed = rows.slice(); render(); }
  };
  socket.onclose = () => { connected = false; clientsAvailable = false; renderClients(); $('connection').textContent = 'Disconnected · reconnecting'; $('connection').className = ''; setTimeout(connect, retry); retry = Math.min(retry*2, 10000); };
  socket.onerror = () => socket.close();
}
$('pause').onclick = () => { paused = !paused; $('pause').setAttribute('aria-pressed', String(paused)); $('pause').querySelector('span').textContent = paused ? 'Resume view' : 'Pause view'; if (!paused) displayed = rows.slice(); render(); };
document.querySelectorAll('[data-filter]').forEach(button => { button.onclick = () => { filter = button.dataset.filter; document.querySelectorAll('[data-filter]').forEach(b => b.setAttribute('aria-pressed', String(b === button))); render(); }; });
$('search').oninput = render;
setInterval(() => { $('clock').textContent = new Date().toLocaleTimeString([], {hour12:false}); const last = rows.at(-1); $('last-event').textContent = last ? `Last source event ${Math.max(0, Math.floor(Date.now()/1000-last.time))}s ago${connected ? '' : ' · disconnected'}` : 'Waiting for the first event'; }, 1000);
connect();
