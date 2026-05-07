// Surriti Visualizer v2 — comprehensive D3 force graph for Surriti's
// SurrealDB-backed temporal knowledge graph.
//
// Renders directional, labeled, parallel-curve edges; surfaces every field
// already on the row (qualifiers, roles, frame metadata, conflict groups,
// lineage); offers purpose-built truth/raw/provenance/conflict/timeline/frame lenses; an
// inline filter sidebar; conflict + timeline panels; episode transcript
// modal; theme, export, and localStorage persistence.

const STORAGE_KEY = "surriti.viz.v2";
const DEBOUNCE_MS = 250;

// ---------- DOM handles ----------
const svg = d3.select("#graph");
const stage = document.querySelector(".stage");
const statusEl = document.querySelector("#status");
const detailsEl = document.querySelector("#details");
const conflictListEl = document.querySelector("#conflictList");
const timelineViewEl = document.querySelector("#timelineView");
const conflictBadgeEl = document.querySelector("#conflictBadge");
const aliasListEl = document.querySelector("#aliasList");const aliasBadgeEl = document.querySelector("#aliasBadge");
const dossierViewEl = document.querySelector("#dossierView");
const frameHealthEl = document.querySelector("#frameHealth");
const selectionKindEl = document.querySelector("#selectionKind");
const transcriptModal = document.querySelector("#transcriptModal");
const transcriptBody = document.querySelector("#transcriptBody");
const exportMenu = document.querySelector("#exportMenu");

const controls = {
  group: document.querySelector("#groupInput"),
  groupList: document.querySelector("#groupList"),
  search: document.querySelector("#searchInput"),
  edgeVisibility: document.querySelector("#edgeVisibility"),
  statusPreset: document.querySelector("#statusPreset"),
  aggregate: document.querySelector("#aggregateToggle"),
  asOfToggle: document.querySelector("#asOfToggle"),
  asOfSlider: document.querySelector("#asOfSlider"),
  asOfLabel: document.querySelector("#asOfLabel"),
  refresh: document.querySelector("#refreshBtn"),
  fit: document.querySelector("#fitBtn"),
  freeze: document.querySelector("#freezeBtn"),
  clear: document.querySelector("#clearBtn"),
  theme: document.querySelector("#themeBtn"),
  export: document.querySelector("#exportBtn"),
  kinds: Array.from(document.querySelectorAll(".kindToggle")),
  sources: Array.from(document.querySelectorAll(".sourceToggle")),
  views: Array.from(document.querySelectorAll('input[name="view"]')),
  frameSearch: document.querySelector("#frameSearch"),
  frameList: document.querySelector("#frameList"),
  minConfidence: document.querySelector("#minConfidence"),
  minConfidenceLabel: document.querySelector("#minConfidenceLabel"),
  validAfter: document.querySelector("#validAfter"),
  validBefore: document.querySelector("#validBefore"),
};

const counts = {
  nodes: document.querySelector("#nodeCount"),
  links: document.querySelector("#linkCount"),
  groups: document.querySelector("#groupCount"),
};

const palette = {
  entity: "var(--entity)",
  episode: "var(--episode)",
  community: "var(--community)",
  relates_to: "var(--fact)",
  mentions: "var(--mention)",
  has_member: "var(--member)",
  invalid: "var(--bad)",
};

// ---------- State ----------
let rawGraph = { nodes: [], links: [], meta: {} };
let timelineBounds = { min: null, max: null };
let frames = [];                          // [{canonical_name, ...}]
let frameByName = new Map();              // lowercased canonical_name -> frame
let selectedFrames = new Set();           // empty = all
let simulation;
let selected = null;                      // { item, type: 'node'|'edge' }
let frozen = false;
let conflictsCache = [];
let egoFocus = null;                      // { nodeId, depth, includeEpisodes, includeInvalidated }

const lensDefaults = {
  truth: { apiView: "entities", edgeVisibility: "active", statusPreset: "current", aggregate: true },
  raw: { apiView: "full", edgeVisibility: "all", statusPreset: "all", aggregate: false },
  provenance: { apiView: "full", edgeVisibility: "all", statusPreset: "all", aggregate: true },
  conflicts: { apiView: "entities", edgeVisibility: "conflicts", statusPreset: "unresolved", aggregate: true },
  timeline: { apiView: "entities", edgeVisibility: "all", statusPreset: "all", aggregate: true },
  frames: { apiView: "entities", edgeVisibility: "all", statusPreset: "all", aggregate: true },
};

// ---------- SVG layers + arrow markers ----------
const root = svg.append("g");
const linkLayer = root.append("g").attr("class", "links");
const labelLayer = root.append("g").attr("class", "edge-labels");
const nodeLayer = root.append("g").attr("class", "nodes");

const defs = svg.append("defs");
function ensureArrowMarker(id, color) {
  if (defs.select(`#${id}`).size()) return;
  defs.append("marker")
    .attr("id", id)
    .attr("viewBox", "-0 -5 10 10")
    .attr("refX", 9)
    .attr("refY", 0)
    .attr("orient", "auto")
    .attr("markerWidth", 6)
    .attr("markerHeight", 6)
    .attr("xoverflow", "visible")
    .append("path")
    .attr("d", "M 0,-5 L 10,0 L 0,5")
    .attr("fill", color)
    .attr("stroke", "none");
}

const zoom = d3.zoom()
  .scaleExtent([0.08, 4])
  .on("zoom", event => {
    root.attr("transform", event.transform);
    updateLabelVisibility(event.transform.k);
    persistState();
  });
svg.call(zoom);
updateLabelVisibility(1);

function dimensions() {
  const rect = stage.getBoundingClientRect();
  return { width: rect.width, height: rect.height };
}

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.style.color = isError ? "var(--bad)" : "";
}

function idOf(value) { return typeof value === "object" ? value.id : value; }

function updateLabelVisibility(k) {
  root
    .classed("zoom-far", k < 0.45)
    .classed("zoom-mid", k >= 0.45 && k < 1.1)
    .classed("zoom-near", k >= 1.1);
}

// ---------- Persistence ----------
function loadPersisted() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    return JSON.parse(raw);
  } catch { return null; }
}
let persistTimer = null;
function persistState() {
  clearTimeout(persistTimer);
  persistTimer = setTimeout(() => {
    const data = {
      view: currentView(),
      group: controls.group.value,
      kinds: controls.kinds.filter(c => c.checked).map(c => c.value),
      statusPreset: controls.statusPreset.value,
      sources: controls.sources.filter(c => c.checked).map(c => c.value),
      edgeVisibility: controls.edgeVisibility.value,
      aggregate: controls.aggregate.checked,
      minConfidence: controls.minConfidence.value,
      validAfter: controls.validAfter.value,
      validBefore: controls.validBefore.value,
      asOfToggle: controls.asOfToggle.checked,
      asOfSlider: controls.asOfSlider.value,
      theme: document.documentElement.dataset.theme,
      selectedFrames: Array.from(selectedFrames),
      selection: selected ? { id: selected.item.id, type: selected.type } : null,
      egoFocus,
    };
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(data)); }
    catch { /* quota exceeded — skip */ }
  }, 200);
}
function applyPersisted(p) {
  if (!p) return;
  if (p.theme) document.documentElement.dataset.theme = p.theme;
  if (p.view) {
    const view = p.view === "full" ? "raw" : p.view === "entities" ? "truth" : p.view === "episodes" ? "provenance" : p.view;
    const r = controls.views.find(r => r.value === view);
    if (r) r.checked = true;
  }
  if (typeof p.group === "string") controls.group.value = p.group;
  if (Array.isArray(p.kinds)) controls.kinds.forEach(c => c.checked = p.kinds.includes(c.value));
  if (Array.isArray(p.sources)) controls.sources.forEach(c => c.checked = p.sources.includes(c.value));
  if (typeof p.statusPreset === "string") controls.statusPreset.value = p.statusPreset;
  else if (Array.isArray(p.statuses)) controls.statusPreset.value = "all";
  if (typeof p.edgeVisibility === "string") controls.edgeVisibility.value = p.edgeVisibility;
  else if (typeof p.invalid === "boolean") controls.edgeVisibility.value = p.invalid ? "all" : "active";
  if (typeof p.aggregate === "boolean") controls.aggregate.checked = p.aggregate;
  if (p.conflictOnly === true) controls.edgeVisibility.value = "conflicts";
  if (p.derivedOnly === true) controls.edgeVisibility.value = "derived";
  if (p.minConfidence != null) {
    controls.minConfidence.value = p.minConfidence;
    updateMinConfidenceLabel();
  }
  if (p.validAfter != null) controls.validAfter.value = p.validAfter;
  if (p.validBefore != null) controls.validBefore.value = p.validBefore;
  if (typeof p.asOfToggle === "boolean") controls.asOfToggle.checked = p.asOfToggle;
  if (p.asOfSlider != null) controls.asOfSlider.value = p.asOfSlider;
  if (Array.isArray(p.selectedFrames)) selectedFrames = new Set(p.selectedFrames);
  if (p.egoFocus) egoFocus = p.egoFocus;
}

// ---------- Theme ----------
function toggleTheme() {
  const cur = document.documentElement.dataset.theme || "dark";
  document.documentElement.dataset.theme = cur === "dark" ? "light" : "dark";
  persistState();
  // Re-resolve CSS variables on link strokes by re-rendering.
  render();
}

// ---------- Filters / payload ----------
function currentView() {
  const checked = controls.views.find(r => r.checked);
  return checked ? checked.value : "truth";
}

function currentLens() {
  return lensDefaults[currentView()] || lensDefaults.truth;
}

function applyLensDefaults(view) {
  const lens = lensDefaults[view] || lensDefaults.truth;
  controls.edgeVisibility.value = lens.edgeVisibility;
  controls.statusPreset.value = lens.statusPreset;
  controls.aggregate.checked = lens.aggregate;
  controls.kinds.forEach(c => {
    c.checked = lens.apiView !== "entities" || c.value !== "episode";
  });
  if (view === "provenance") {
    controls.kinds.forEach(c => c.checked = c.value !== "community");
  }
}

function buildQueryParams() {
  const p = new URLSearchParams();
  const lens = currentLens();
  p.set("view", currentView());
  const group = controls.group.value.trim();
  if (group) p.set("group_id", group);
  const edgeVisibility = controls.edgeVisibility.value;
  p.set("include_invalid", edgeVisibility === "all" || edgeVisibility === "invalidated" || currentView() === "raw" ? "true" : "false");
  p.set("aggregate_mentions", controls.aggregate.checked ? "true" : "false");
  if (edgeVisibility === "conflicts") p.set("edge_visibility", "conflicts");
  else if (edgeVisibility === "derived") p.set("edge_visibility", "derived");
  else if (edgeVisibility === "non_derived") p.set("edge_visibility", "non_derived");
  else if (edgeVisibility === "invalidated") p.set("edge_visibility", "invalidated");
  else if (edgeVisibility === "active") p.set("edge_visibility", "active");
  for (const status of statusesForPreset(controls.statusPreset.value)) p.append("status", status);
  const checkedSources = controls.sources.filter(c => c.checked);
  if (checkedSources.length < controls.sources.length) {
    for (const c of checkedSources) p.append("source_type", c.value);
  }
  for (const fname of selectedFrames) p.append("canonical_name", fname);
  const minConf = Number(controls.minConfidence.value) / 100;
  if (minConf > 0) p.set("min_confidence", minConf.toFixed(2));
  if (controls.validAfter.value) p.set("valid_after", new Date(controls.validAfter.value).toISOString());
  if (controls.validBefore.value) p.set("valid_before", new Date(controls.validBefore.value).toISOString());
  if (controls.asOfToggle.checked) {
    const iso = sliderToIso();
    if (iso) p.set("as_of", iso);
  }
  return p;
}

function statusesForPreset(preset) {
  if (preset === "current") return ["active"];
  if (preset === "historical") return ["superseded"];
  if (preset === "unresolved") return ["needs_resolution"];
  return [];
}

function visibleKinds() {
  return new Set(controls.kinds.filter(input => input.checked).map(input => input.value));
}

function textForNode(node) {
  return [
    node.name, node.label, node.summary, node.content || "",
    node.group_id, JSON.stringify(node.attributes || {}),
  ].join(" ").toLowerCase();
}
function textForLink(link) {
  return [
    link.name, link.label, link.canonical_name, link.fact, link.group_id,
    JSON.stringify(link.qualifiers || {}),
    JSON.stringify(link.roles || {}),
    JSON.stringify(link.attributes || {}),
  ].join(" ").toLowerCase();
}

function filteredGraph() {
  const kinds = visibleKinds();
  const query = controls.search.value.trim().toLowerCase();
  const nodeById = new Map(rawGraph.nodes.map(n => [n.id, n]));
  let nodes = rawGraph.nodes.filter(n => kinds.has(n.kind));
  let links = rawGraph.links.filter(l =>
    nodeById.has(idOf(l.source)) && nodeById.has(idOf(l.target))
  );

  if (query) {
    const matchingNodeIds = new Set(nodes.filter(n => textForNode(n).includes(query)).map(n => n.id));
    const matchingLinks = links.filter(l => textForLink(l).includes(query));
    for (const l of matchingLinks) {
      matchingNodeIds.add(idOf(l.source));
      matchingNodeIds.add(idOf(l.target));
    }
    nodes = nodes.filter(n => matchingNodeIds.has(n.id));
  }
  const allowed = new Set(nodes.map(n => n.id));
  links = links.filter(l => allowed.has(idOf(l.source)) && allowed.has(idOf(l.target)));

  if (egoFocus && egoFocus.nodeId && allowed.has(egoFocus.nodeId)) {
    const focusIds = egoNodeIds(egoFocus.nodeId, links, Number(egoFocus.depth || 1));
    nodes = nodes.filter(n => focusIds.has(n.id));
    links = links.filter(l => {
      const source = idOf(l.source);
      const target = idOf(l.target);
      if (!focusIds.has(source) || !focusIds.has(target)) return false;
      const sourceNode = nodeById.get(source);
      const targetNode = nodeById.get(target);
      if (!egoFocus.includeEpisodes && ((sourceNode && sourceNode.kind === "episode") || (targetNode && targetNode.kind === "episode"))) return false;
      if (!egoFocus.includeInvalidated && (l.invalid_at || l.expired_at || l.status === "superseded")) return false;
      return true;
    });
  }

  // Pair-bucket parallel edges for curve offset; preserves direction so
  // A->B and B->A get opposite-side curves.
  const bucket = new Map();
  for (const l of links) {
    const a = idOf(l.source), b = idOf(l.target);
    const key = a < b ? `${a}|${b}` : `${b}|${a}`;
    if (!bucket.has(key)) bucket.set(key, []);
    bucket.get(key).push(l);
  }
  for (const [, edges] of bucket) {
    edges.forEach((e, i) => {
      const n = edges.length;
      // For 1 edge: curve = 0. For multiple: fan out.
      e._curve = n === 1 ? 0 : (i - (n - 1) / 2) * 18;
      // Reverse curve sign when this edge runs against the canonical pair direction.
      const a = idOf(e.source), b = idOf(e.target);
      if (a > b) e._curve = -e._curve;
    });
  }

  return { nodes: nodes.map(n => ({ ...n })), links: links.map(l => ({ ...l })) };
}

function egoNodeIds(startId, links, depth) {
  const adjacency = new Map();
  for (const l of links) {
    const a = idOf(l.source);
    const b = idOf(l.target);
    if (!adjacency.has(a)) adjacency.set(a, new Set());
    if (!adjacency.has(b)) adjacency.set(b, new Set());
    adjacency.get(a).add(b);
    adjacency.get(b).add(a);
  }
  const seen = new Set([startId]);
  let frontier = new Set([startId]);
  for (let i = 0; i < depth; i += 1) {
    const next = new Set();
    for (const id of frontier) {
      for (const neighbor of adjacency.get(id) || []) {
        if (!seen.has(neighbor)) {
          seen.add(neighbor);
          next.add(neighbor);
        }
      }
    }
    frontier = next;
  }
  return seen;
}

// ---------- Render geometry ----------
function nodeRadius(node) {
  const degree = Number(node.degree || 0);
  if (node.kind === "episode") return Math.min(18, 8 + degree * 0.7);
  if (node.kind === "community") return Math.min(26, 13 + degree * 0.9);
  return Math.min(24, 10 + degree * 0.85);
}

function linkColorFor(link) {
  if (link.status === "needs_resolution") return "var(--bad)";
  if (link.invalid_at || link.expired_at) return "var(--bad)";
  return palette[link.kind] || palette.relates_to;
}

function linkClassesFor(link) {
  const cls = ["link"];
  if (link.aggregated) cls.push("aggregated");
  if (link.invalid_at || link.expired_at) cls.push("invalidated");
  if (link.status === "superseded") cls.push("superseded");
  if (link.status === "needs_resolution") cls.push("conflict");
  if (link.derived) cls.push("derived");
  return cls.join(" ");
}

function linkWidth(link) {
  if (link.kind === "mentions") return 1.2;
  if (link.kind === "has_member") return 1.8;
  return Math.min(5, 1.5 + ((link.episodes && link.episodes.length) || 0) * 0.6);
}

function linkLabel(link) {
  const base = link.canonical_name || link.label || link.name || link.kind;
  if (link.aggregated && link.count > 1) return `mentions ×${link.count}`;
  // Inline qualifier values on top of the predicate.
  const qvals = link.qualifiers && typeof link.qualifiers === "object"
    ? Object.values(link.qualifiers).filter(v => v !== null && v !== undefined && v !== "")
    : [];
  const tail = qvals.length ? ` · ${qvals.slice(0, 2).map(v => String(v)).join(" · ")}` : "";
  let text = base + tail;
  if (link.status === "needs_resolution") text = "⚠ " + text;
  return text.length > 32 ? text.slice(0, 31) + "…" : text;
}

function bezierPath(d) {
  const sx = d.source.x, sy = d.source.y;
  const tx = d.target.x, ty = d.target.y;
  const dx = tx - sx, dy = ty - sy;
  const dist = Math.sqrt(dx * dx + dy * dy) || 1;
  const off = d._curve || 0;
  // Perpendicular offset for the bezier control point.
  const mx = (sx + tx) / 2 - dy * off / dist;
  const my = (sy + ty) / 2 + dx * off / dist;
  // Pull line endpoints back to circle edge so arrowheads sit cleanly.
  const tr = nodeRadius(d.target) + 4;
  const sr = nodeRadius(d.source) + 1;
  const ang1 = Math.atan2(my - sy, mx - sx);
  const ang2 = Math.atan2(ty - my, tx - mx);
  const sx2 = sx + Math.cos(ang1) * sr;
  const sy2 = sy + Math.sin(ang1) * sr;
  const tx2 = tx - Math.cos(ang2) * tr;
  const ty2 = ty - Math.sin(ang2) * tr;
  return `M${sx2},${sy2} Q${mx},${my} ${tx2},${ty2}`;
}

function labelTransform(d) {
  const sx = d.source.x, sy = d.source.y;
  const tx = d.target.x, ty = d.target.y;
  const dx = tx - sx, dy = ty - sy;
  const dist = Math.sqrt(dx * dx + dy * dy) || 1;
  const off = d._curve || 0;
  const mx = (sx + tx) / 2 - dy * off / dist;
  const my = (sy + ty) / 2 + dx * off / dist;
  let angle = Math.atan2(ty - sy, tx - sx) * 180 / Math.PI;
  if (angle > 90) angle -= 180;
  else if (angle < -90) angle += 180;
  return `translate(${mx},${my}) rotate(${angle})`;
}

// Determine if the frame for this edge marks it symmetric — skip arrowhead.
function isSymmetric(link) {
  const f = frameByName.get((link.canonical_name || link.name || "").toLowerCase());
  return f && f.directionality === "symmetric";
}

// ---------- Render ----------
function render() {
  const graph = filteredGraph();
  const { width, height } = dimensions();
  counts.nodes.textContent = graph.nodes.length;
  counts.links.textContent = graph.links.length;
  counts.groups.textContent = new Set(rawGraph.nodes.map(n => n.group_id).filter(Boolean)).size;

  svg.attr("viewBox", [0, 0, width, height]);
  if (simulation) simulation.stop();

  // Ensure arrow markers exist for each link kind (color resolved at use site).
  ensureArrowMarker("arrow-relates_to", "currentColor");
  ensureArrowMarker("arrow-mentions", "currentColor");
  ensureArrowMarker("arrow-has_member", "currentColor");
  ensureArrowMarker("arrow-bad", "currentColor");

  const links = linkLayer.selectAll("path")
    .data(graph.links, d => d.id)
    .join("path")
    .attr("class", linkClassesFor)
    .attr("stroke", linkColorFor)
    .attr("stroke-width", linkWidth)
    .attr("marker-end", d => isSymmetric(d) ? null
      : (d.status === "needs_resolution" ? "url(#arrow-bad)" : `url(#arrow-${d.kind})`))
    .style("color", linkColorFor)
    .on("click", (event, d) => { event.stopPropagation(); selectItem(d, "edge"); });

  const labels = labelLayer.selectAll("text")
    .data(
      graph.links.filter(l => l.kind !== "has_member" || l.aggregated),
      d => d.id,
    )
    .join("text")
    .attr("class", d => [
      "link-label",
      d.kind === "mentions" ? "mentions" : "",
      d.kind === "mentions" || d.kind === "has_member" ? "low-importance" : "",
      d.status === "needs_resolution" ? "conflict" : "",
      d.aggregated ? "mention-count" : "",
    ].filter(Boolean).join(" "))
    .attr("dy", "-0.45em")
    .text(linkLabel);

  // Conflict-flag map driven by conflict_group_id.
  const conflictByNode = new Map();
  const mentionByNode = new Map();
  const historicalOnly = new Set();
  const activeByNode = new Map();
  const possibleAliases = aliasGroups(graph.nodes);
  for (const l of graph.links) {
    if (l.kind === "mentions") {
      const tid = idOf(l.target);
      mentionByNode.set(tid, (mentionByNode.get(tid) || 0) + Number(l.count || 1));
    }
    if (l.kind === "relates_to") {
      const sid = idOf(l.source);
      if (l.conflict_group_id && l.status === "needs_resolution") conflictByNode.set(sid, (conflictByNode.get(sid) || 0) + 1);
      if (l.status === "active" && !l.invalid_at && !l.expired_at) activeByNode.set(sid, (activeByNode.get(sid) || 0) + 1);
      if (l.status === "superseded" || l.invalid_at || l.expired_at) historicalOnly.add(sid);
    }
  }
  for (const id of activeByNode.keys()) historicalOnly.delete(id);
  const aliasNodeIds = new Set(possibleAliases.flatMap(g => g.nodes.map(n => n.id)));

  const nodes = nodeLayer.selectAll("g")
    .data(graph.nodes, d => d.id)
    .join(
      enter => {
        const g = enter.append("g").attr("class", "node").call(dragBehavior());
        g.append("circle");
        g.append("text").attr("dy", "0.35em").attr("x", d => nodeRadius(d) + 6);
        g.append("text").attr("class", "quality-badge").attr("text-anchor", "middle").attr("dy", "0.35em");
        return g;
      },
      update => update,
      exit => exit.remove()
    )
    .classed("major", d => Number(d.degree || 0) >= 4 || d.kind === "community")
    .classed("conflict", d => (conflictByNode.get(d.id) || 0) > 0)
    .on("click", (event, d) => { event.stopPropagation(); selectItem(d, "node"); });

  nodes.select("circle")
    .attr("r", nodeRadius)
    .attr("fill", d => `var(--${d.kind})`)
    .attr("fill-opacity", d => d.kind === "episode" ? 0.78 : 0.92);

  nodes.selectAll("text").filter(function () { return !this.classList.contains("quality-badge"); })
    .text(d => d.name || d.id);

  nodes.select("text.quality-badge")
    .attr("x", d => nodeRadius(d) + 4)
    .attr("y", d => -nodeRadius(d) - 4)
    .text(d => qualityBadge(d, { conflictByNode, mentionByNode, historicalOnly, aliasNodeIds }));

  simulation = d3.forceSimulation(graph.nodes)
    .force("link", d3.forceLink(graph.links).id(d => d.id).distance(linkDistance).strength(0.42))
    .force("charge", d3.forceManyBody().strength(d => d.kind === "episode" ? -130 : -260))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collide", d3.forceCollide().radius(d => nodeRadius(d) + 12).iterations(2))
    .force("x", d3.forceX(width / 2).strength(0.035))
    .force("y", d3.forceY(height / 2).strength(0.035));

  // rAF-throttled tick: at most one DOM update per animation frame.
  let pending = false;
  simulation.on("tick", () => {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => {
      links.attr("d", bezierPath);
      labels.attr("transform", labelTransform);
      nodes.attr("transform", d => `translate(${d.x},${d.y})`);
      pending = false;
    });
  });

  if (frozen) simulation.stop();
  updateFocus();
  renderAliasPanel();
  renderFrameHealth();
  setStatus(`${graph.nodes.length} nodes · ${graph.links.length} edges · view=${currentView()}`);
}

function qualityBadge(node, indexes) {
  if ((indexes.conflictByNode.get(node.id) || 0) > 0) return "!";
  if (indexes.aliasNodeIds.has(node.id)) return "?";
  if (indexes.historicalOnly.has(node.id)) return "H";
  if ((indexes.mentionByNode.get(node.id) || 0) >= 5) return String(indexes.mentionByNode.get(node.id));
  return "";
}

function linkDistance(link) {
  if (link.kind === "mentions") return 86;
  if (link.kind === "has_member") return 118;
  return 142;
}

function dragBehavior() {
  return d3.drag()
    .on("start", (event, d) => {
      if (!event.active && simulation) simulation.alphaTarget(0.25).restart();
      d.fx = d.x; d.fy = d.y;
    })
    .on("drag", (event, d) => { d.fx = event.x; d.fy = event.y; })
    .on("end", (event, d) => {
      if (!event.active && simulation) simulation.alphaTarget(0);
      if (!frozen) { d.fx = null; d.fy = null; }
    });
}

// ---------- Selection / focus ----------
function selectItem(item, type) {
  selected = { item, type };
  renderDetails(item, type);
  switchTab("inspector");
  if (type === "node" && item.kind === "entity") loadTimeline(item.id);
  if (type === "node" && item.kind === "entity") loadDossier(item.id);
  updateFocus();
  persistState();
}
function clearSelection() {
  selected = null;
  selectionKindEl.textContent = "Nothing selected";
  detailsEl.innerHTML = `<p class="empty">Select a node or relationship to inspect every field already on the row.</p>`;
  timelineViewEl.innerHTML = `<p class="empty">Select an entity to see its temporal history.</p>`;
  updateFocus();
  persistState();
}

function updateFocus() {
  const selectedId = selected && selected.type === "node" ? selected.item.id : null;
  const selectedLinkId = selected && selected.type === "edge" ? selected.item.id : null;
  const related = new Set();
  if (selectedId) {
    related.add(selectedId);
    for (const l of rawGraph.links) {
      if (idOf(l.source) === selectedId || idOf(l.target) === selectedId) {
        related.add(idOf(l.source));
        related.add(idOf(l.target));
      }
    }
  }
  nodeLayer.selectAll("g")
    .classed("selected", d => selectedId === d.id)
    .classed("dimmed", d => selectedId && !related.has(d.id));
  linkLayer.selectAll("path")
    .classed("dimmed", d => selectedId && idOf(d.source) !== selectedId && idOf(d.target) !== selectedId)
    .classed("selected", d => selectedLinkId === d.id);
  labelLayer.selectAll("text")
    .classed("dimmed", d => selectedId && idOf(d.source) !== selectedId && idOf(d.target) !== selectedId);
}

// ---------- Inspector renderers ----------
function renderDetails(item, type) {
  if (type === "node") renderNodeDetails(item);
  else renderEdgeDetails(item);
}

function renderNodeDetails(node) {
  selectionKindEl.textContent = node.kind;
  const incident = rawGraph.links.filter(l => idOf(l.source) === node.id || idOf(l.target) === node.id);
  const nodeById = new Map(rawGraph.nodes.map(n => [n.id, n]));
  const neighborIds = new Set(incident.map(l => idOf(l.source) === node.id ? idOf(l.target) : idOf(l.source)));
  const neighbors = rawGraph.nodes.filter(n => neighborIds.has(n.id)).slice(0, 30);

  const outFacts = incident.filter(l => l.kind === "relates_to" && idOf(l.source) === node.id);
  const current = outFacts.filter(l => !l.invalid_at && !l.expired_at && l.status !== "superseded");
  const former = outFacts.filter(l => l.invalid_at || l.expired_at || l.status === "superseded");
  const conflicts = current.filter(l => l.status === "needs_resolution");

  const factRow = (link, cls) => {
    const target = nodeById.get(idOf(link.target));
    const label = link.canonical_name || link.name;
    return `<div class="fact-row ${cls}" data-edge-id="${escapeAttr(link.id)}">
      <span class="pill clickable">${escapeHtml(label)}</span>
      <span class="obj">${escapeHtml((target && target.name) || idOf(link.target))}</span>
    </div>`;
  };

  detailsEl.innerHTML = `
    <section class="detail-block">
      <h2>${escapeHtml(node.name || node.id)}${conflicts.length ? ` <span class="pill bad">CONFLICT</span>` : ""}</h2>
      <div class="pill-row">${(node.labels || [node.kind]).map(l => `<span class="pill">${escapeHtml(l)}</span>`).join("")}</div>
      ${node.summary ? `<p>${escapeHtml(node.summary)}</p>` : ""}
    </section>
    <section class="detail-block">
      <h3>Focus</h3>
      <div class="focus-controls">
        <label>Depth
          <select id="egoDepth">
            <option value="1" ${egoFocus && egoFocus.depth === 1 ? "selected" : ""}>1</option>
            <option value="2" ${egoFocus && egoFocus.depth === 2 ? "selected" : ""}>2</option>
            <option value="3" ${egoFocus && egoFocus.depth === 3 ? "selected" : ""}>3</option>
          </select>
        </label>
        <label>Episodes
          <select id="egoEpisodes">
            <option value="false" ${egoFocus && egoFocus.includeEpisodes === false ? "selected" : ""}>No</option>
            <option value="true" ${egoFocus && egoFocus.includeEpisodes ? "selected" : ""}>Yes</option>
          </select>
        </label>
        <label>Invalidated
          <select id="egoInvalidated">
            <option value="false" ${egoFocus && egoFocus.includeInvalidated === false ? "selected" : ""}>No</option>
            <option value="true" ${egoFocus && egoFocus.includeInvalidated ? "selected" : ""}>Yes</option>
          </select>
        </label>
      </div>
      <div class="action-row">
        <button type="button" data-focus-node="${escapeAttr(node.id)}">Focus on this node</button>
        ${egoFocus ? `<button type="button" data-clear-focus="true">Clear focus</button>` : ""}
      </div>
    </section>
    <section class="detail-block">
      <h3>Properties</h3>
      ${kv({
        uuid: node.uuid,
        group: node.group_id || "default",
        degree: node.degree || 0,
        source: node.source || "",
        reference: node.reference_time || "",
        created: node.created_at || "",
      })}
    </section>
    ${conflicts.length ? `<section class="detail-block">
      <h3>Conflicts</h3>${conflicts.map(l => factRow(l, "conflict")).join("")}
    </section>` : ""}
    ${current.length ? `<section class="detail-block">
      <h3>Current facts</h3>${current.filter(l => !conflicts.includes(l)).map(l => factRow(l, "current")).join("")}
    </section>` : ""}
    ${former.length ? `<section class="detail-block">
      <h3>Former facts</h3>${former.slice(0, 50).map(l => factRow(l, "former")).join("")}
    </section>` : ""}
    <section class="detail-block">
      <h3>Other relationships</h3>
      ${incident.filter(l => l.kind !== "relates_to").length
        ? `<div class="pill-row">${incident.filter(l => l.kind !== "relates_to").slice(0, 18).map(l => `<span class="pill">${escapeHtml(l.label || l.name)}</span>`).join("")}</div>`
        : `<p class="empty">None.</p>`}
    </section>
    ${neighbors.length ? `<section class="detail-block">
      <h3>Neighbors</h3>
      <div class="pill-row">${neighbors.map(n => `<span class="pill clickable" data-node-id="${escapeAttr(n.id)}">${escapeHtml(n.name)}</span>`).join("")}</div>
    </section>` : ""}
    <section class="detail-block">
      <h3>Raw</h3>
      <pre>${escapeHtml(JSON.stringify(node.raw || node, null, 2))}</pre>
    </section>
  `;
  bindInspectorClicks();
}

function renderEdgeDetails(edge) {
  selectionKindEl.textContent = edge.kind;
  const nodeById = new Map(rawGraph.nodes.map(n => [n.id, n]));
  const source = nodeById.get(idOf(edge.source));
  const target = nodeById.get(idOf(edge.target));
  const frame = frameByName.get((edge.canonical_name || edge.name || "").toLowerCase());

  const block = (title, body) => body ? `<section class="detail-block"><h3>${escapeHtml(title)}</h3>${body}</section>` : "";

  const identity = kv({
    name: edge.name || "",
    canonical: edge.canonical_name || "",
    domain: edge.domain || "",
    fact_key: edge.fact_key || "",
  });

  const frameBadge = frame ? `<div class="pill-row">
    <span class="pill frame">${escapeHtml(frame.canonical_name)}</span>
    <span class="pill">dir: ${escapeHtml(frame.directionality)}</span>
    <span class="pill">temp: ${escapeHtml(frame.temporal_kind)}</span>
    <span class="pill">card: ${escapeHtml(frame.cardinality)}</span>
    <span class="pill">policy: ${escapeHtml(frame.contradiction_policy)}</span>
  </div>` : "";

  const qualifiers = edge.qualifiers && Object.keys(edge.qualifiers).length
    ? kv(edge.qualifiers) : "";
  const roles = edge.roles && Object.keys(edge.roles).length
    ? kv(edge.roles) : "";

  const status = kv({
    status: edge.status || "",
    polarity: edge.polarity || "",
    confidence: edge.confidence == null ? "" : Number(edge.confidence).toFixed(2),
    singleton: edge.singleton ? "true" : "",
    temporal: edge.temporal ? "true" : "",
    derived: edge.derived ? "true" : "",
    source_type: edge.source_type || "",
  });

  const temporal = kv({
    valid: edge.valid_at || "",
    invalid: edge.invalid_at || "",
    expired: edge.expired_at || "",
    created: edge.created_at || "",
  });

  const provenance = edge.episodes && edge.episodes.length
    ? `<div class="chip-row">${edge.episodes.map(uid => `<span class="chip" data-episode="${escapeAttr(uid)}">${escapeHtml(shortUuid(uid))}</span>`).join("")}</div>`
    : "";

  const lineage = ((edge.supersedes && edge.supersedes.length) || edge.superseded_by || edge.derived_from)
    ? `<div class="chip-row">
        ${edge.supersedes.map(uid => `<span class="chip" data-edge="${escapeAttr(uid)}" title="supersedes">⊐ ${escapeHtml(shortUuid(uid))}</span>`).join("")}
        ${edge.superseded_by ? `<span class="chip" data-edge="${escapeAttr(edge.superseded_by)}" title="superseded by">⊏ ${escapeHtml(shortUuid(edge.superseded_by))}</span>` : ""}
        ${edge.derived_from ? `<span class="chip" data-edge="${escapeAttr(edge.derived_from)}" title="derived from">↩ ${escapeHtml(shortUuid(edge.derived_from))}</span>` : ""}
      </div>` : "";

  const conflict = edge.conflict_group_id
    ? `<button type="button" data-cg="${escapeAttr(edge.conflict_group_id)}">Open conflict group</button>` : "";

  detailsEl.innerHTML = `
    <section class="detail-block">
      <h2>${escapeHtml(edge.canonical_name || edge.name || edge.kind)}</h2>
      ${edge.fact ? `<p>${escapeHtml(edge.fact)}</p>` : ""}
      <div class="pill-row">
        <span class="pill clickable" data-node-id="${escapeAttr(idOf(edge.source))}">${escapeHtml((source && source.name) || idOf(edge.source))}</span>
        <span class="muted">→</span>
        <span class="pill clickable" data-node-id="${escapeAttr(idOf(edge.target))}">${escapeHtml((target && target.name) || idOf(edge.target))}</span>
      </div>
    </section>
    ${block("Identity", identity)}
    ${block("Frame", frameBadge)}
    ${block("Qualifiers", qualifiers)}
    ${block("Roles", roles)}
    ${block("Status", status)}
    ${block("Temporal", temporal)}
    ${block("Provenance (episodes)", provenance)}
    ${block("Lineage", lineage)}
    ${block("Conflict", conflict)}
    ${block("Attributes", Object.keys(edge.attributes || {}).length ? kv(edge.attributes) : "")}
    <section class="detail-block">
      <h3>Raw</h3>
      <pre>${escapeHtml(JSON.stringify(edge.raw || edge, null, 2))}</pre>
    </section>
  `;
  bindInspectorClicks();
}

function bindInspectorClicks() {
  detailsEl.querySelectorAll("[data-edge-id]").forEach(el => {
    el.addEventListener("click", () => {
      const id = el.dataset.edgeId;
      const link = rawGraph.links.find(l => l.id === id);
      if (link) selectItem(link, "edge");
    });
  });
  detailsEl.querySelectorAll("[data-edge]").forEach(el => {
    el.addEventListener("click", () => {
      const id = el.dataset.edge;
      const link = rawGraph.links.find(l => l.id === id || l.uuid === id);
      if (link) selectItem(link, "edge");
    });
  });
  detailsEl.querySelectorAll("[data-node-id]").forEach(el => {
    el.addEventListener("click", () => {
      const id = el.dataset.nodeId;
      const node = rawGraph.nodes.find(n => n.id === id);
      if (node) selectItem(node, "node");
    });
  });
  detailsEl.querySelectorAll("[data-episode]").forEach(el => {
    el.addEventListener("click", () => openTranscript(el.dataset.episode));
  });
  detailsEl.querySelectorAll("[data-cg]").forEach(el => {
    el.addEventListener("click", () => {
      switchTab("conflicts");
      highlightConflictGroup(el.dataset.cg);
    });
  });
  detailsEl.querySelectorAll("[data-focus-node]").forEach(el => {
    el.addEventListener("click", () => {
      egoFocus = {
        nodeId: el.dataset.focusNode,
        depth: Number((document.querySelector("#egoDepth") || {}).value || 1),
        includeEpisodes: (document.querySelector("#egoEpisodes") || {}).value === "true",
        includeInvalidated: (document.querySelector("#egoInvalidated") || {}).value === "true",
      };
      render();
      persistState();
    });
  });
  detailsEl.querySelectorAll("[data-clear-focus]").forEach(el => {
    el.addEventListener("click", () => {
      egoFocus = null;
      render();
      persistState();
    });
  });
}

function kv(data) {
  const rows = Object.entries(data).filter(([, v]) => v !== "" && v !== null && v !== undefined);
  if (!rows.length) return "";
  return `<dl class="kv">${rows.map(([k, v]) =>
    `<dt>${escapeHtml(k)}</dt><dd>${escapeHtml(typeof v === "object" ? JSON.stringify(v) : String(v))}</dd>`
  ).join("")}</dl>`;
}

function shortUuid(u) {
  const s = String(u);
  return s.length > 8 ? s.slice(0, 8) : s;
}

function escapeHtml(value) {
  return String(value == null ? "" : value)
    .split("&").join("&amp;").split("<").join("&lt;").split(">").join("&gt;")
    .split('"').join("&quot;").split("'").join("&#039;");
}
function escapeAttr(value) { return escapeHtml(value); }

// ---------- Quality panels ----------
function normalizeName(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/^the\s+/, "")
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function levenshtein(a, b) {
  if (a === b) return 0;
  if (!a.length) return b.length;
  if (!b.length) return a.length;
  const row = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i += 1) {
    let prev = row[0];
    row[0] = i;
    for (let j = 1; j <= b.length; j += 1) {
      const tmp = row[j];
      row[j] = Math.min(
        row[j] + 1,
        row[j - 1] + 1,
        prev + (a[i - 1] === b[j - 1] ? 0 : 1),
      );
      prev = tmp;
    }
  }
  return row[b.length];
}

function aliasGroups(nodes = rawGraph.nodes) {
  const entities = nodes.filter(n => n.kind === "entity" && n.name);
  const groups = [];
  const used = new Set();
  for (let i = 0; i < entities.length; i += 1) {
    const a = entities[i];
    if (used.has(a.id)) continue;
    const an = normalizeName(a.name);
    if (!an) continue;
    const members = [a];
    for (let j = i + 1; j < entities.length; j += 1) {
      const b = entities[j];
      if (used.has(b.id)) continue;
      const bn = normalizeName(b.name);
      if (!bn) continue;
      const close = an === bn
        || an.includes(bn)
        || bn.includes(an)
        || levenshtein(an, bn) <= Math.max(1, Math.floor(Math.min(an.length, bn.length) * 0.22));
      if (close) members.push(b);
    }
    if (members.length > 1) {
      members.forEach(n => used.add(n.id));
      groups.push({ key: an, nodes: members });
    }
  }
  return groups;
}

function renderAliasPanel() {
  const groups = aliasGroups(rawGraph.nodes);
  aliasBadgeEl.classList.toggle("hidden", groups.length === 0);
  aliasBadgeEl.textContent = groups.length;
  if (!groups.length) {
    aliasListEl.innerHTML = `<p class="empty">No likely aliases found.</p>`;
    return;
  }
  aliasListEl.innerHTML = groups.map((g, i) => `
    <article class="alias-card">
      <header class="card-head">
        <strong>${escapeHtml(g.nodes.map(n => n.name).join(" / "))}</strong>
        <span class="pill">case/name match</span>
      </header>
      <div class="pill-row">
        ${g.nodes.map(n => `<span class="pill clickable" data-node-id="${escapeAttr(n.id)}">${escapeHtml(n.name)}</span>`).join("")}
      </div>
      <div class="action-row">
        <button type="button" data-copy-merge="${i}">Copy merge command</button>
        <button type="button" data-open-aliases="${i}">Open nodes</button>
      </div>
    </article>
  `).join("");
  aliasListEl.querySelectorAll("[data-node-id]").forEach(el => {
    el.addEventListener("click", () => {
      const node = rawGraph.nodes.find(n => n.id === el.dataset.nodeId);
      if (node) selectItem(node, "node");
    });
  });
  aliasListEl.querySelectorAll("[data-copy-merge]").forEach(el => {
    el.addEventListener("click", async () => {
      const group = groups[Number(el.dataset.copyMerge)];
      const canonical = group.nodes[0];
      const aliases = group.nodes.slice(1);
      const command = `surriti repair merge-entities --canonical ${canonical.id} ${aliases.map(n => `--alias ${n.id}`).join(" ")}`;
      try { await navigator.clipboard.writeText(command); el.textContent = "Copied"; }
      catch { el.textContent = command; }
    });
  });
  aliasListEl.querySelectorAll("[data-open-aliases]").forEach(el => {
    el.addEventListener("click", () => {
      const group = groups[Number(el.dataset.openAliases)];
      controls.search.value = group.nodes.map(n => n.name).join(" ");
      egoFocus = null;
      render();
    });
  });
}

function renderFrameHealth() {
  const stats = new Map();
  for (const l of rawGraph.links) {
    if (l.kind !== "relates_to") continue;
    const name = l.canonical_name || l.name || "unknown/custom";
    const row = stats.get(name) || { count: 0, active: 0, superseded: 0, conflicts: 0, unknown: false };
    row.count += 1;
    if (l.status === "active" && !l.invalid_at && !l.expired_at) row.active += 1;
    if (l.status === "superseded" || l.invalid_at || l.expired_at) row.superseded += 1;
    if (l.status === "needs_resolution" || l.conflict_group_id) row.conflicts += 1;
    const frame = frameByName.get(name.toLowerCase());
    row.unknown = row.unknown || !frame || frame.directionality === "unknown";
    stats.set(name, row);
  }
  const rows = Array.from(stats.entries()).sort((a, b) => b[1].count - a[1].count);
  if (!rows.length) {
    frameHealthEl.innerHTML = `<p class="empty">No relation frames in this graph slice.</p>`;
    return;
  }
  frameHealthEl.innerHTML = `
    <section class="detail-block">
      <h3>Frame Health</h3>
      <table class="frame-health-table">
        <thead><tr><th>Frame</th><th>Count</th><th>Active</th><th>Superseded</th><th>Conflicts</th><th>Unknown</th></tr></thead>
        <tbody>${rows.map(([name, s]) => `
          <tr>
            <td><button type="button" data-frame-filter="${escapeAttr(name)}">${escapeHtml(name)}</button></td>
            <td>${s.count}</td><td>${s.active}</td><td>${s.superseded}</td><td>${s.conflicts}</td><td>${s.unknown ? "yes" : "no"}</td>
          </tr>`).join("")}</tbody>
      </table>
    </section>`;
  frameHealthEl.querySelectorAll("[data-frame-filter]").forEach(el => {
    el.addEventListener("click", () => {
      selectedFrames = new Set([el.dataset.frameFilter]);
      renderFrameList();
      const r = controls.views.find(r => r.value === "frames");
      if (r) r.checked = true;
      scheduleReload();
      switchTab("inspector");
    });
  });
}

// ---------- Conflicts tab ----------
async function loadConflicts() {
  const params = new URLSearchParams();
  const group = controls.group.value.trim();
  if (group) params.set("group_id", group);
  try {
    const res = await fetch(`/api/conflicts?${params}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    conflictsCache = data.groups || [];
    renderConflictPanel();
  } catch (err) {
    conflictListEl.innerHTML = `<p class="empty">Failed to load conflicts: ${escapeHtml(err.message)}</p>`;
  }
}
function renderConflictPanel() {
  const n = conflictsCache.length;
  conflictBadgeEl.classList.toggle("hidden", n === 0);
  conflictBadgeEl.textContent = n;
  if (!n) {
    conflictListEl.innerHTML = `<p class="empty">No unresolved conflicts.</p>`;
    return;
  }
  conflictListEl.innerHTML = conflictsCache.map(g => `
    <article class="conflict-card bad" data-cg="${escapeAttr(g.conflict_group_id)}">
      <header class="card-head">
        <strong>${escapeHtml((g.subject && g.subject.name) || "subject")}</strong>
        <span class="pill frame">${escapeHtml(g.canonical_name)}</span>
      </header>
      <div class="chip-row">
        ${g.edges.map(e => `<span class="chip" data-edge="${escapeAttr(e.id)}" title="${escapeAttr(e.fact || "")}">${escapeHtml(e.canonical_name || e.name)} → ${escapeHtml(shortUuid(e.target))}</span>`).join("")}
      </div>
      <div class="muted" style="font-size:11px">cg: ${escapeHtml(g.conflict_group_id)}</div>
    </article>
  `).join("");
  conflictListEl.querySelectorAll("[data-edge]").forEach(el => {
    el.addEventListener("click", () => {
      const id = el.dataset.edge;
      const link = rawGraph.links.find(l => l.id === id || l.uuid === id);
      if (link) selectItem(link, "edge");
    });
  });
}
function highlightConflictGroup(cg) {
  const card = conflictListEl.querySelector(`[data-cg="${cg}"]`);
  if (card) card.scrollIntoView({ behavior: "smooth", block: "center" });
}

// ---------- Timeline tab ----------
async function loadTimeline(uuid) {
  const params = new URLSearchParams();
  const group = controls.group.value.trim();
  if (group) params.set("group_id", group);
  timelineViewEl.innerHTML = `<p class="empty">Loading timeline…</p>`;
  try {
    const res = await fetch(`/api/entity/${encodeURIComponent(uuid)}/timeline?${params}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderTimeline(data);
  } catch (err) {
    timelineViewEl.innerHTML = `<p class="empty">Failed to load timeline: ${escapeHtml(err.message)}</p>`;
  }
}
// ---------- Dossier tab ----------
async function loadDossier(uuid) {
  if (!dossierViewEl) return;
  const params = new URLSearchParams();
  const group = controls.group.value.trim();
  if (group) params.set("group_id", group);
  dossierViewEl.innerHTML = `<p class="empty">Loading dossier…</p>`;
  try {
    const res = await fetch(`/api/entity/${encodeURIComponent(uuid)}/profile?${params}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    renderDossier(await res.json());
  } catch (err) {
    dossierViewEl.innerHTML = `<p class="empty">Failed to load dossier: ${escapeHtml(err.message)}</p>`;
  }
}

function renderDossier(data) {
  const aliases = (data.aliases || []).filter(a => a && a !== data.canonical_name);
  const aliasChips = aliases.length
    ? `<div class="pill-row">${aliases.map(a => `<span class="pill">${escapeHtml(a)}</span>`).join("")}</div>`
    : `<p class="empty">No alias variants recorded.</p>`;
  const facts = (data.facts || []).slice(0, 30).map(f => `
    <div class="fact-row">
      <span class="pill">${escapeHtml(f.canonical_name || f.name || "relates_to")}</span>
      <span class="obj">${escapeHtml(f.fact || "")}</span>
    </div>
  `).join("");
  const salience = typeof data.salience === "number" ? data.salience.toFixed(2) : data.salience;
  dossierViewEl.innerHTML = `
    <section class="detail-block">
      <h2>${escapeHtml(data.canonical_name || data.name || "Entity")}</h2>
      <div class="pill-row">
        <span class="pill">salience ${salience}</span>
        <span class="pill">mentions ${data.mention_count || 0}</span>
        ${data.last_seen_at ? `<span class="pill">last seen ${escapeHtml(String(data.last_seen_at))}</span>` : ""}
      </div>
      ${data.profile_summary ? `<p>${escapeHtml(data.profile_summary)}</p>` : `<p class="empty">No dossier summary yet.</p>`}
    </section>
    <section class="detail-block">
      <h3>Aliases</h3>
      ${aliasChips}
    </section>
    <section class="detail-block">
      <h3>Recent facts</h3>
      ${facts || `<p class="empty">No facts recorded.</p>`}
    </section>
  `;
}

function renderTimeline(data) {
  const events = data.events || [];
  if (!events.length) {
    timelineViewEl.innerHTML = `<p class="empty">No facts recorded for ${escapeHtml((data.subject && data.subject.name) || "this entity")}.</p>`;
    return;
  }  // Build supersedes index for chain rendering.
  const byUuid = new Map(events.map(e => [e.uuid, e]));
  const cards = events.map((e, i) => {
    const cls = timelineClass(e);
    const when = e.valid_at || e.created_at || "";
    const status = e.status === "needs_resolution" ? `<span class="pill bad">⚠</span>` : "";
    const lineage = e.supersedes && e.supersedes.length
      ? `<div class="chain-arrow">superseded ${e.supersedes.map(uid => escapeHtml(shortUuid(uid))).join(", ")}</div>`
      : "";
    return `<article class="timeline-card ${cls}" data-edge="${escapeAttr(e.id)}">
      <header class="card-head">
        <span class="pred">${escapeHtml(e.canonical_name || e.name)} -> ${escapeHtml((e.target && e.target.name) || shortUuid((e.target && e.target.uuid) || ""))}</span>
        ${status}
      </header>
      <div class="when">${escapeHtml(when)}${e.invalid_at ? ` -> ${escapeHtml(e.invalid_at)}` : ""}</div>
      ${e.episodes && e.episodes.length ? `<div class="chip-row">${e.episodes.map(uid => `<span class="chip" data-episode="${escapeAttr(uid)}">${escapeHtml(shortUuid(uid))}</span>`).join("")}</div>` : ""}
      ${lineage}
    </article>`;
  });
  timelineViewEl.innerHTML = `
    <section class="detail-block">
      <h3>${escapeHtml((data.subject && data.subject.name) || "Entity")} timeline · ${events.length}</h3>
      ${cards.join('<div class="chain-arrow">↓</div>')}
    </section>`;
  timelineViewEl.querySelectorAll("[data-edge]").forEach(el => {
    el.addEventListener("click", e => {
      if (e.target.closest("[data-episode]")) return;
      const id = el.dataset.edge;
      const link = rawGraph.links.find(l => l.id === id || l.uuid === id);
      if (link) selectItem(link, "edge");
    });
  });
  timelineViewEl.querySelectorAll("[data-episode]").forEach(el => {
    el.addEventListener("click", e => { e.stopPropagation(); openTranscript(el.dataset.episode); });
  });
}

function timelineClass(edge) {
  if (edge.status === "needs_resolution" || edge.conflict_group_id) return "conflict";
  if (edge.derived) return "derived";
  if (edge.status === "superseded" || edge.invalid_at || edge.expired_at) return "superseded";
  return "active";
}

// ---------- Transcript modal ----------
async function openTranscript(uuid) {
  transcriptBody.innerHTML = `<p class="empty">Loading…</p>`;
  if (!transcriptModal.open) transcriptModal.showModal();
  try {
    const params = new URLSearchParams();
    const group = controls.group.value.trim();
    if (group) params.set("group_id", group);
    const res = await fetch(`/api/episode/${encodeURIComponent(uuid)}?${params}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const ep = await res.json();
    const mentioned = rawGraph.links
      .filter(l => l.kind === "mentions" && idOf(l.source) === uuid)
      .map(l => rawGraph.nodes.find(n => n.id === idOf(l.target)))
      .filter(Boolean);
    const createdFacts = rawGraph.links
      .filter(l => l.kind === "relates_to" && (l.episodes || []).includes(uuid));
    const invalidatedFacts = createdFacts.filter(l => l.invalid_at || l.expired_at || l.status === "superseded");
    transcriptBody.innerHTML = `
      <h2>${escapeHtml(ep.name || "Episode")}</h2>
      ${kv({
        uuid: ep.uuid,
        source: ep.source || "",
        source_description: ep.source_description || "",
        reference_time: ep.reference_time || "",
        created_at: ep.created_at || "",
        mentions: ep.mention_count == null ? 0 : ep.mention_count,
      })}
      <h3>Entities Mentioned</h3>
      ${mentioned.length ? `<div class="pill-row">${mentioned.map(n => `<span class="pill">${escapeHtml(n.name || n.id)}</span>`).join("")}</div>` : `<p class="empty">Not present in current graph slice.</p>`}
      <h3>Facts Extracted</h3>
      ${createdFacts.length ? `<div class="chip-row">${createdFacts.map(l => `<span class="chip" data-edge="${escapeAttr(l.id)}">${escapeHtml(l.canonical_name || l.name)} -> ${escapeHtml(shortUuid(idOf(l.target)))}</span>`).join("")}</div>` : `<p class="empty">Not present in current graph slice.</p>`}
      <h3>Invalidated / Superseded</h3>
      ${invalidatedFacts.length ? `<div class="chip-row">${invalidatedFacts.map(l => `<span class="chip" data-edge="${escapeAttr(l.id)}">${escapeHtml(l.canonical_name || l.name)} -> ${escapeHtml(shortUuid(idOf(l.target)))}</span>`).join("")}</div>` : `<p class="empty">None in current graph slice.</p>`}
      <h3>Content</h3>
      <pre>${escapeHtml(ep.content || "")}</pre>
    `;
    transcriptBody.querySelectorAll("[data-edge]").forEach(el => {
      el.addEventListener("click", () => {
        const link = rawGraph.links.find(l => l.id === el.dataset.edge);
        if (link) {
          transcriptModal.close();
          selectItem(link, "edge");
        }
      });
    });
  } catch (err) {
    transcriptBody.innerHTML = `<p class="empty">Failed to load: ${escapeHtml(err.message)}</p>`;
  }
}

// ---------- Tabs ----------
function switchTab(name) {
  document.querySelectorAll(".tab").forEach(t => {
    const active = t.dataset.tab === name;
    t.classList.toggle("active", active);
    t.setAttribute("aria-selected", active ? "true" : "false");
  });
  document.querySelectorAll(".tab-panel").forEach(p => {
    p.classList.toggle("active", p.dataset.panel === name);
  });
}

// ---------- Frames ----------
async function loadFrames() {
  const params = new URLSearchParams();
  const group = controls.group.value.trim();
  if (group) params.set("group_id", group);
  try {
    const res = await fetch(`/api/frames?${params}`);
    if (!res.ok) return;
    const data = await res.json();
    frames = data.frames || [];
    frameByName = new Map(frames.map(f => [f.canonical_name.toLowerCase(), f]));
    renderFrameList();
  } catch (err) { console.warn("frames failed", err); }
}
function renderFrameList() {
  const q = (controls.frameSearch.value || "").trim().toLowerCase();
  const filtered = q ? frames.filter(f => f.canonical_name.toLowerCase().includes(q)) : frames;
  controls.frameList.innerHTML = filtered.map(f => {
    const name = f.canonical_name;
    const checked = selectedFrames.has(name) ? "checked" : "";
    return `<label title="${escapeAttr(f.directionality + " · " + f.cardinality + " · " + f.contradiction_policy)}">
      <input type="checkbox" data-frame="${escapeAttr(name)}" ${checked}> ${escapeHtml(name)}
      <span class="muted" style="font-size:11px">${escapeHtml(f.source)}</span>
    </label>`;
  }).join("") || `<p class="empty">No frames.</p>`;
  controls.frameList.querySelectorAll("input[type=checkbox]").forEach(cb => {
    cb.addEventListener("change", () => {
      const name = cb.dataset.frame;
      if (cb.checked) selectedFrames.add(name); else selectedFrames.delete(name);
      scheduleReload();
      persistState();
    });
  });
}

// ---------- Loaders ----------
async function loadGraph() {
  setStatus("Loading graph…");
  const res = await fetch(`/api/graph?${buildQueryParams()}`);
  if (!res.ok) {
    setStatus(await res.text() || `HTTP ${res.status}`, true);
    return;
  }
  rawGraph = await res.json();
  if (rawGraph.meta && rawGraph.meta.group_id) {
    // no-op: group is user-driven.
  }
  render();
  // Re-apply selection if the same item is still on the graph.
  const persisted = loadPersisted();
  if (persisted && persisted.selection && !selected) {
    const sel = persisted.selection;
    if (sel.type === "node") {
      const n = rawGraph.nodes.find(n => n.id === sel.id);
      if (n) selectItem(n, "node");
    } else {
      const l = rawGraph.links.find(l => l.id === sel.id);
      if (l) selectItem(l, "edge");
    }
  } else if (!selected) {
    clearSelection();
  } else {
    renderDetails(selected.item, selected.type);
  }
}
async function loadGroups() {
  try {
    const res = await fetch("/api/groups");
    if (!res.ok) return;
    const data = await res.json();
    controls.groupList.innerHTML = (data.groups || [])
      .map(g => `<option value="${escapeAttr(g.group_id)}">`).join("");
  } catch { /* non-fatal */ }
}
async function loadTimelineBounds() {
  const params = new URLSearchParams();
  const group = controls.group.value.trim();
  if (group) params.set("group_id", group);
  try {
    const res = await fetch(`/api/timeline_bounds?${params}`);
    if (!res.ok) return;
    timelineBounds = await res.json();
    updateAsOfLabel();
  } catch (err) { console.warn("timeline_bounds failed", err); }
}

function sliderToIso() {
  if (!timelineBounds.min || !timelineBounds.max) return null;
  const min = Date.parse(timelineBounds.min);
  const max = Date.parse(timelineBounds.max);
  if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) return null;
  const pct = Number(controls.asOfSlider.value) / 100;
  return new Date(min + pct * (max - min)).toISOString();
}
function updateAsOfLabel() {
  if (!controls.asOfToggle.checked) {
    controls.asOfLabel.textContent = "live";
    controls.asOfSlider.disabled = true;
    return;
  }
  controls.asOfSlider.disabled = !(timelineBounds.min && timelineBounds.max);
  const iso = sliderToIso();
  controls.asOfLabel.textContent = iso ? iso.replace("T", " ").replace(/\.\d+Z?$/, "") : "—";
}
function updateMinConfidenceLabel() {
  const v = Number(controls.minConfidence.value) / 100;
  controls.minConfidenceLabel.textContent = `≥ ${v.toFixed(2)}`;
}

function fitToView() {
  const { width, height } = dimensions();
  const nodes = nodeLayer.selectAll("g").data();
  if (!nodes.length) return;
  const xs = nodes.map(d => d.x), ys = nodes.map(d => d.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const gw = Math.max(1, maxX - minX), gh = Math.max(1, maxY - minY);
  const scale = Math.min(2.2, 0.86 / Math.max(gw / width, gh / height));
  const tx = width / 2 - scale * (minX + gw / 2);
  const ty = height / 2 - scale * (minY + gh / 2);
  svg.transition().duration(450).call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
}

// ---------- Reload scheduling ----------
let reloadTimer = null;
function scheduleReload() {
  clearTimeout(reloadTimer);
  reloadTimer = setTimeout(() => {
    loadGraph().catch(err => setStatus(err.message, true));
    loadConflicts();
  }, DEBOUNCE_MS);
}

// ---------- Export ----------
function downloadBlob(name, blob) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 200);
}
function exportJSON() {
  downloadBlob("surriti-graph.json",
    new Blob([JSON.stringify(rawGraph, null, 2)], { type: "application/json" }));
}
function inlineComputedStyles(svgEl) {
  const clone = svgEl.cloneNode(true);
  // Inline computed styles for export portability.
  const orig = svgEl.querySelectorAll("*");
  const cloneEls = clone.querySelectorAll("*");
  orig.forEach((el, i) => {
    const cs = getComputedStyle(el);
    const props = ["fill", "stroke", "stroke-width", "stroke-dasharray", "opacity",
      "font-size", "font-weight", "font-family"];
    cloneEls[i].setAttribute("style",
      props.map(p => `${p}:${cs.getPropertyValue(p)}`).join(";"));
  });
  return clone;
}
function exportSVG() {
  const svgEl = document.querySelector("#graph");
  const clone = inlineComputedStyles(svgEl);
  const xml = new XMLSerializer().serializeToString(clone);
  downloadBlob("surriti-graph.svg",
    new Blob([`<?xml version="1.0" standalone="no"?>\n${xml}`], { type: "image/svg+xml" }));
}
function exportPNG() {
  const svgEl = document.querySelector("#graph");
  const { width, height } = dimensions();
  const clone = inlineComputedStyles(svgEl);
  clone.setAttribute("width", width);
  clone.setAttribute("height", height);
  const xml = new XMLSerializer().serializeToString(clone);
  const blob = new Blob([xml], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const img = new Image();
  img.onload = () => {
    const canvas = document.createElement("canvas");
    canvas.width = width * 2;
    canvas.height = height * 2;
    const ctx = canvas.getContext("2d");
    const bg = getComputedStyle(document.body).backgroundColor;
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(b => downloadBlob("surriti-graph.png", b), "image/png");
    URL.revokeObjectURL(url);
  };
  img.onerror = () => URL.revokeObjectURL(url);
  img.src = url;
}

// ---------- Wiring ----------
function debounce(fn, ms) {
  let t = null;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

controls.refresh.addEventListener("click", () => loadGraph().catch(err => setStatus(err.message, true)));
controls.fit.addEventListener("click", fitToView);
controls.freeze.addEventListener("click", () => {
  frozen = !frozen;
  controls.freeze.textContent = frozen ? "Thaw" : "Freeze";
  if (frozen && simulation) simulation.stop();
  if (!frozen && simulation) simulation.alpha(0.35).restart();
});
controls.clear.addEventListener("click", clearSelection);
controls.search.addEventListener("input", debounce(render, 200));
controls.edgeVisibility.addEventListener("change", scheduleReload);
controls.statusPreset.addEventListener("change", scheduleReload);
controls.aggregate.addEventListener("change", scheduleReload);
controls.sources.forEach(c => c.addEventListener("change", scheduleReload));
controls.views.forEach(r => r.addEventListener("change", () => {
  applyLensDefaults(currentView());
  egoFocus = null;
  scheduleReload();
  persistState();
}));
controls.minConfidence.addEventListener("input", () => { updateMinConfidenceLabel(); persistState(); });
controls.minConfidence.addEventListener("change", scheduleReload);
controls.validAfter.addEventListener("change", scheduleReload);
controls.validBefore.addEventListener("change", scheduleReload);
controls.frameSearch.addEventListener("input", debounce(renderFrameList, 150));
controls.asOfToggle.addEventListener("change", () => {
  updateAsOfLabel();
  scheduleReload();
});
controls.asOfSlider.addEventListener("input", () => {
  updateAsOfLabel();
  scheduleReload();
});
controls.group.addEventListener("change", () => {
  loadTimelineBounds();
  loadFrames();
  scheduleReload();
  loadConflicts();
});
controls.kinds.forEach(input => input.addEventListener("change", () => { render(); persistState(); }));
controls.theme.addEventListener("click", toggleTheme);
controls.export.addEventListener("click", e => {
  e.stopPropagation();
  exportMenu.classList.toggle("hidden");
});
exportMenu.addEventListener("click", e => {
  const which = e.target && e.target.dataset && e.target.dataset.export;
  if (which === "json") exportJSON();
  else if (which === "svg") exportSVG();
  else if (which === "png") exportPNG();
  exportMenu.classList.add("hidden");
});
document.addEventListener("click", e => {
  if (!exportMenu.contains(e.target) && e.target !== controls.export) exportMenu.classList.add("hidden");
});

document.querySelectorAll(".tab").forEach(t => t.addEventListener("click", () => switchTab(t.dataset.tab)));

document.querySelector("#transcriptClose").addEventListener("click", () => transcriptModal.close());
transcriptModal.addEventListener("click", e => {
  if (e.target === transcriptModal) transcriptModal.close();
});

svg.on("click", clearSelection);
window.addEventListener("resize", debounce(render, 100));

// Keyboard shortcuts.
document.addEventListener("keydown", e => {
  if (e.target.matches("input, textarea, select")) return;
  if (e.key === "/") { e.preventDefault(); controls.search.focus(); }
  else if (e.key === "Escape") clearSelection();
  else if (e.key === " ") { e.preventDefault(); controls.freeze.click(); }
  else if (e.key === "f") fitToView();
  else if (e.key === "r") loadGraph().catch(err => setStatus(err.message, true));
  else if (e.key === "t") toggleTheme();
  else if (e.key >= "1" && e.key <= "6") {
    const order = ["truth", "raw", "provenance", "conflicts", "timeline", "frames"];
    const r = controls.views.find(r => r.value === order[Number(e.key) - 1]);
    if (r) { r.checked = true; applyLensDefaults(currentView()); scheduleReload(); }
  }
});

// ---------- Boot ----------
applyPersisted(loadPersisted());
updateMinConfidenceLabel();
loadGroups();
loadTimelineBounds();
loadFrames();
loadConflicts();
loadGraph()
  .then(() => setTimeout(fitToView, 650))
  .catch(err => setStatus(err.message, true));
