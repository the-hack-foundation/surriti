const svg = d3.select("#graph");
const stage = document.querySelector(".stage");
const statusEl = document.querySelector("#status");
const detailsEl = document.querySelector("#details");
const selectionKindEl = document.querySelector("#selectionKind");

const controls = {
  group: document.querySelector("#groupInput"),
  search: document.querySelector("#searchInput"),
  invalid: document.querySelector("#invalidToggle"),
  aggregate: document.querySelector("#aggregateToggle"),
  asOfToggle: document.querySelector("#asOfToggle"),
  asOfSlider: document.querySelector("#asOfSlider"),
  asOfLabel: document.querySelector("#asOfLabel"),
  refresh: document.querySelector("#refreshBtn"),
  fit: document.querySelector("#fitBtn"),
  freeze: document.querySelector("#freezeBtn"),
  clear: document.querySelector("#clearBtn"),
  kinds: Array.from(document.querySelectorAll(".kindToggle")),
};

let timelineBounds = { min: null, max: null };

const counts = {
  nodes: document.querySelector("#nodeCount"),
  links: document.querySelector("#linkCount"),
  groups: document.querySelector("#groupCount"),
};

const palette = {
  entity: "#55c7a5",
  episode: "#f0b65a",
  community: "#d06aa1",
  relates_to: "#8bb8ff",
  mentions: "#d9cf75",
  has_member: "#95d66f",
  invalid: "#e56c60",
};

let rawGraph = { nodes: [], links: [], meta: {} };
let simulation;
let selected = null;
let frozen = false;

const root = svg.append("g");
const linkLayer = root.append("g").attr("class", "links");
const labelLayer = root.append("g").attr("class", "edge-labels");
const nodeLayer = root.append("g").attr("class", "nodes");

const zoom = d3.zoom()
  .scaleExtent([0.08, 4])
  .on("zoom", event => root.attr("transform", event.transform));

svg.call(zoom);

function dimensions() {
  const rect = stage.getBoundingClientRect();
  return { width: rect.width, height: rect.height };
}

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.style.color = isError ? "#e56c60" : "";
}

function kindColor(kind) {
  return palette[kind] || palette.entity;
}

function linkColor(link) {
  if (link.invalid_at || link.expired_at) return palette.invalid;
  return palette[link.kind] || palette.relates_to;
}

function nodeRadius(node) {
  const degree = Number(node.degree || 0);
  if (node.kind === "episode") return Math.min(18, 8 + degree * 0.7);
  if (node.kind === "community") return Math.min(26, 13 + degree * 0.9);
  return Math.min(24, 10 + degree * 0.85);
}

function linkWidth(link) {
  if (link.kind === "mentions") return 1.2;
  if (link.kind === "has_member") return 1.8;
  return Math.min(5, 1.5 + (link.episodes?.length || 0) * 0.6);
}

function visibleKinds() {
  return new Set(controls.kinds.filter(input => input.checked).map(input => input.value));
}

function textForNode(node) {
  return [
    node.name,
    node.label,
    node.summary,
    node.content,
    node.group_id,
    JSON.stringify(node.attributes || {}),
  ].join(" ").toLowerCase();
}

function textForLink(link) {
  return [
    link.name,
    link.fact,
    link.group_id,
    JSON.stringify(link.attributes || {}),
  ].join(" ").toLowerCase();
}

function filteredGraph() {
  const kinds = visibleKinds();
  const query = controls.search.value.trim().toLowerCase();
  const nodeById = new Map(rawGraph.nodes.map(node => [node.id, node]));
  let nodes = rawGraph.nodes.filter(node => kinds.has(node.kind));
  let links = rawGraph.links.filter(link => {
    if (!controls.invalid.checked && (link.invalid_at || link.expired_at)) return false;
    return nodeById.has(idOf(link.source)) && nodeById.has(idOf(link.target));
  });

  if (query) {
    const matchingNodes = new Set(nodes.filter(node => textForNode(node).includes(query)).map(node => node.id));
    const matchingLinks = links.filter(link => textForLink(link).includes(query));
    for (const link of matchingLinks) {
      matchingNodes.add(idOf(link.source));
      matchingNodes.add(idOf(link.target));
    }
    nodes = nodes.filter(node => matchingNodes.has(node.id));
    const allowed = new Set(nodes.map(node => node.id));
    links = links.filter(link =>
      allowed.has(idOf(link.source)) &&
      allowed.has(idOf(link.target)) &&
      (matchingLinks.includes(link) || matchingNodes.has(idOf(link.source)) || matchingNodes.has(idOf(link.target)))
    );
  } else {
    const allowed = new Set(nodes.map(node => node.id));
    links = links.filter(link => allowed.has(idOf(link.source)) && allowed.has(idOf(link.target)));
  }

  return { nodes: nodes.map(node => ({ ...node })), links: links.map(link => ({ ...link })) };
}

function idOf(value) {
  return typeof value === "object" ? value.id : value;
}

function render() {
  const graph = filteredGraph();
  const { width, height } = dimensions();

  counts.nodes.textContent = graph.nodes.length;
  counts.links.textContent = graph.links.length;
  counts.groups.textContent = new Set(rawGraph.nodes.map(node => node.group_id).filter(Boolean)).size;

  svg.attr("viewBox", [0, 0, width, height]);

  if (simulation) simulation.stop();

  const links = linkLayer.selectAll("line")
    .data(graph.links, d => d.id)
    .join(
      enter => enter.append("line").attr("class", "link"),
      update => update,
      exit => exit.remove()
    )
    .attr("stroke", linkColor)
    .attr("stroke-width", linkWidth)
    .classed("invalidated", d => Boolean(d.invalid_at || d.expired_at))
    .classed("aggregated", d => Boolean(d.aggregated))
    .on("click", (event, d) => {
      event.stopPropagation();
      selectItem(d, "edge");
    });

  const labels = labelLayer.selectAll("text")
    .data(
      graph.links.filter(link => link.kind === "relates_to" || (link.aggregated && link.count > 1)),
      d => d.id,
    )
    .join("text")
    .attr("class", d => d.aggregated ? "link-label mention-count" : "link-label")
    .text(d => d.aggregated ? `×${d.count}` : d.name);

  // Conflict detection: an entity has a singleton conflict when more than
  // one currently-active relates_to edge shares the same predicate name.
  // The badge keeps the visualizer close to raw data — it just surfaces a
  // condition already present on the edges.
  const conflictByNode = new Map();
  for (const node of graph.nodes) conflictByNode.set(node.id, 0);
  const grouped = new Map();
  for (const link of graph.links) {
    if (link.kind !== "relates_to") continue;
    if (link.invalid_at || link.expired_at) continue;
    if (!link.singleton) continue;
    const key = `${idOf(link.source)}::${(link.name || "").toLowerCase()}`;
    if (!grouped.has(key)) grouped.set(key, new Set());
    grouped.get(key).add(idOf(link.target));
  }
  for (const [key, targets] of grouped.entries()) {
    if (targets.size <= 1) continue;
    const sourceId = key.split("::")[0];
    conflictByNode.set(sourceId, (conflictByNode.get(sourceId) || 0) + 1);
  }

  const nodes = nodeLayer.selectAll("g")
    .data(graph.nodes, d => d.id)
    .join(
      enter => {
        const g = enter.append("g").attr("class", "node").call(dragBehavior());
        g.append("circle");
        g.append("text").attr("dy", "0.35em").attr("x", d => nodeRadius(d) + 6);
        g.append("text").attr("class", "conflict-badge").attr("text-anchor", "middle").attr("dy", "0.35em");
        return g;
      },
      update => update,
      exit => exit.remove()
    )
    .classed("conflict", d => (conflictByNode.get(d.id) || 0) > 0)
    .on("click", (event, d) => {
      event.stopPropagation();
      selectItem(d, "node");
    });

  nodes.select("circle")
    .attr("r", nodeRadius)
    .attr("fill", d => kindColor(d.kind))
    .attr("fill-opacity", d => d.kind === "episode" ? 0.78 : 0.92);

  nodes.select("text").filter(function () { return !this.classList.contains("conflict-badge"); })
    .text(d => d.name || d.id);

  nodes.select("text.conflict-badge")
    .attr("x", d => nodeRadius(d) + 4)
    .attr("y", d => -nodeRadius(d) - 4)
    .text(d => (conflictByNode.get(d.id) || 0) > 0 ? "⚠" : "");

  simulation = d3.forceSimulation(graph.nodes)
    .force("link", d3.forceLink(graph.links).id(d => d.id).distance(linkDistance).strength(0.42))
    .force("charge", d3.forceManyBody().strength(d => d.kind === "episode" ? -130 : -260))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collide", d3.forceCollide().radius(d => nodeRadius(d) + 12).iterations(2))
    .force("x", d3.forceX(width / 2).strength(0.035))
    .force("y", d3.forceY(height / 2).strength(0.035))
    .on("tick", () => {
      links
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);

      labels
        .attr("x", d => (d.source.x + d.target.x) / 2)
        .attr("y", d => (d.source.y + d.target.y) / 2);

      nodes.attr("transform", d => `translate(${d.x},${d.y})`);
    });

  if (frozen) simulation.stop();
  updateFocus();
  setStatus(`${graph.nodes.length} nodes, ${graph.links.length} relationships`);
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
      d.fx = d.x;
      d.fy = d.y;
    })
    .on("drag", (event, d) => {
      d.fx = event.x;
      d.fy = event.y;
    })
    .on("end", (event, d) => {
      if (!event.active && simulation) simulation.alphaTarget(0);
      if (!frozen) {
        d.fx = null;
        d.fy = null;
      }
    });
}

function selectItem(item, type) {
  selected = { item, type };
  renderDetails(item, type);
  updateFocus();
}

function clearSelection() {
  selected = null;
  selectionKindEl.textContent = "Nothing selected";
  detailsEl.innerHTML = `<p class="empty">Select a node or relationship to inspect its facts, timestamps, attributes, and neighbors.</p>`;
  updateFocus();
}

function updateFocus() {
  const graph = filteredGraph();
  const selectedId = selected?.type === "node" ? selected.item.id : null;
  const selectedLinkId = selected?.type === "edge" ? selected.item.id : null;
  const related = new Set();
  if (selectedId) {
    related.add(selectedId);
    for (const link of graph.links) {
      if (idOf(link.source) === selectedId || idOf(link.target) === selectedId) {
        related.add(idOf(link.source));
        related.add(idOf(link.target));
      }
    }
  }
  nodeLayer.selectAll("g")
    .classed("selected", d => selectedId === d.id)
    .classed("dimmed", d => selectedId && !related.has(d.id));
  linkLayer.selectAll("line")
    .classed("dimmed", d => selectedId && idOf(d.source) !== selectedId && idOf(d.target) !== selectedId)
    .classed("selected", d => selectedLinkId === d.id);
  labelLayer.selectAll("text")
    .classed("dimmed", d => selectedId && idOf(d.source) !== selectedId && idOf(d.target) !== selectedId);
}

function renderDetails(item, type) {
  if (type === "node") renderNodeDetails(item);
  else renderEdgeDetails(item);
}

function renderNodeDetails(node) {
  selectionKindEl.textContent = node.kind;
  const graph = filteredGraph();
  const incident = graph.links.filter(link => idOf(link.source) === node.id || idOf(link.target) === node.id);
  const neighborIds = new Set(incident.map(link => idOf(link.source) === node.id ? idOf(link.target) : idOf(link.source)));
  const neighbors = graph.nodes.filter(n => neighborIds.has(n.id)).slice(0, 30);

  // Partition outgoing relates_to edges into current / former / conflicts
  // so the inspector mirrors the temporal state of the raw edges without
  // additional dressup.
  const nodeById = new Map(graph.nodes.map(n => [n.id, n]));
  const outFacts = incident.filter(l => l.kind === "relates_to" && idOf(l.source) === node.id);
  const current = outFacts.filter(l => !l.invalid_at && !l.expired_at);
  const former = outFacts.filter(l => l.invalid_at || l.expired_at);
  const grouped = new Map();
  for (const link of current) {
    if (!link.singleton) continue;
    const key = (link.name || "").toLowerCase();
    if (!grouped.has(key)) grouped.set(key, []);
    grouped.get(key).push(link);
  }
  const conflictKeys = new Set(
    Array.from(grouped.entries())
      .filter(([, links]) => new Set(links.map(l => idOf(l.target))).size > 1)
      .map(([key]) => key),
  );

  const factRow = (link, cls) => {
    const target = nodeById.get(idOf(link.target));
    return `<div class="fact-row ${cls}"><span class="pill">${escapeHtml(link.name)}</span><span class="obj">${escapeHtml(target?.name || idOf(link.target))}</span></div>`;
  };

  const currentNonConflict = current.filter(l => !conflictKeys.has((l.name || "").toLowerCase()));
  const conflictRows = current.filter(l => conflictKeys.has((l.name || "").toLowerCase()));

  detailsEl.innerHTML = `
    <section class="detail-block">
      <h2>${escapeHtml(node.name || node.id)}${conflictKeys.size ? ` <span class="badge">CONFLICT</span>` : ""}</h2>
      <div class="pill-row">${(node.labels || [node.kind]).map(label => `<span class="pill">${escapeHtml(label)}</span>`).join("")}</div>
      ${node.summary ? `<p>${escapeHtml(node.summary)}</p>` : ""}
      ${node.content ? `<p class="muted">${escapeHtml(truncate(node.content, 320))}</p>` : ""}
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
    ${conflictRows.length ? `<section class="detail-block">
      <h3>Conflicts</h3>
      ${conflictRows.map(l => factRow(l, "conflict")).join("")}
    </section>` : ""}
    ${currentNonConflict.length ? `<section class="detail-block">
      <h3>Current facts</h3>
      ${currentNonConflict.map(l => factRow(l, "current")).join("")}
    </section>` : ""}
    ${former.length ? `<section class="detail-block">
      <h3>Former facts</h3>
      ${former.slice(0, 50).map(l => factRow(l, "former")).join("")}
    </section>` : ""}
    <section class="detail-block">
      <h3>Other relationships</h3>
      ${incident.filter(l => l.kind !== "relates_to").length
        ? incident.filter(l => l.kind !== "relates_to").slice(0, 18).map(edgeSummary).join("")
        : `<p class="empty">None.</p>`}
    </section>
    <section class="detail-block">
      <h3>Neighbors</h3>
      ${neighbors.length ? `<div class="pill-row">${neighbors.map(n => `<span class="pill">${escapeHtml(n.name)}</span>`).join("")}</div>` : `<p class="empty">No visible neighbors.</p>`}
    </section>
    <section class="detail-block">
      <h3>Raw</h3>
      <pre>${escapeHtml(JSON.stringify(node.raw || node, null, 2))}</pre>
    </section>
  `;
}

function renderEdgeDetails(edge) {
  selectionKindEl.textContent = edge.kind;
  const nodeById = new Map(rawGraph.nodes.map(node => [node.id, node]));
  const source = nodeById.get(idOf(edge.source));
  const target = nodeById.get(idOf(edge.target));

  detailsEl.innerHTML = `
    <section class="detail-block">
      <h2>${escapeHtml(edge.name || edge.kind)}</h2>
      ${edge.fact ? `<p>${escapeHtml(edge.fact)}</p>` : ""}
      <div class="pill-row">
        <span class="pill">${escapeHtml(source?.name || idOf(edge.source))}</span>
        <span class="pill">${escapeHtml(target?.name || idOf(edge.target))}</span>
      </div>
    </section>
    <section class="detail-block">
      <h3>Temporal</h3>
      ${kv({
        valid: edge.valid_at || "",
        invalid: edge.invalid_at || "",
        expired: edge.expired_at || "",
        created: edge.created_at || "",
        episodes: edge.episodes?.length || 0,
      })}
    </section>
    <section class="detail-block">
      <h3>Attributes</h3>
      <pre>${escapeHtml(JSON.stringify(edge.attributes || {}, null, 2))}</pre>
    </section>
    <section class="detail-block">
      <h3>Raw</h3>
      <pre>${escapeHtml(JSON.stringify(edge.raw || edge, null, 2))}</pre>
    </section>
  `;
}

function edgeSummary(edge) {
  const nodeById = new Map(rawGraph.nodes.map(node => [node.id, node]));
  const source = nodeById.get(idOf(edge.source));
  const target = nodeById.get(idOf(edge.target));
  return `
    <div class="kv">
      <dt>${escapeHtml(edge.name)}</dt>
      <dd>${escapeHtml(source?.name || idOf(edge.source))} → ${escapeHtml(target?.name || idOf(edge.target))}${edge.invalid_at ? " (invalidated)" : ""}</dd>
      ${edge.fact ? `<dt>fact</dt><dd>${escapeHtml(edge.fact)}</dd>` : ""}
    </div>
  `;
}

function kv(data) {
  const rows = Object.entries(data).filter(([, value]) => value !== "");
  if (!rows.length) return `<p class="empty">No properties.</p>`;
  return `<dl class="kv">${rows.map(([key, value]) => `<dt>${escapeHtml(key)}</dt><dd>${escapeHtml(String(value))}</dd>`).join("")}</dl>`;
}

function truncate(text, max) {
  return text.length > max ? `${text.slice(0, max - 1)}...` : text;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function loadGraph() {
  setStatus("Loading graph...");
  const params = new URLSearchParams();
  const group = controls.group.value.trim();
  if (group) params.set("group_id", group);
  params.set("include_invalid", controls.invalid.checked ? "true" : "false");
  params.set("aggregate_mentions", controls.aggregate.checked ? "true" : "false");
  if (controls.asOfToggle.checked) {
    const iso = sliderToIso();
    if (iso) params.set("as_of", iso);
  }
  const response = await fetch(`/api/graph?${params.toString()}`);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }
  rawGraph = await response.json();
  selected = null;
  render();
  clearSelection();
}

async function loadTimelineBounds() {
  const params = new URLSearchParams();
  const group = controls.group.value.trim();
  if (group) params.set("group_id", group);
  try {
    const response = await fetch(`/api/timeline_bounds?${params.toString()}`);
    if (!response.ok) return;
    timelineBounds = await response.json();
    updateAsOfLabel();
  } catch (err) {
    // Non-fatal: slider just stays disabled.
    console.warn("timeline_bounds failed", err);
  }
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

function fitToView() {
  const { width, height } = dimensions();
  const nodes = nodeLayer.selectAll("g").data();
  if (!nodes.length) return;
  const xs = nodes.map(d => d.x);
  const ys = nodes.map(d => d.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const graphWidth = Math.max(1, maxX - minX);
  const graphHeight = Math.max(1, maxY - minY);
  const scale = Math.min(2.2, 0.86 / Math.max(graphWidth / width, graphHeight / height));
  const tx = width / 2 - scale * (minX + graphWidth / 2);
  const ty = height / 2 - scale * (minY + graphHeight / 2);
  svg.transition().duration(450).call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
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
controls.search.addEventListener("input", render);
controls.invalid.addEventListener("change", () => loadGraph().catch(err => setStatus(err.message, true)));
controls.aggregate.addEventListener("change", () => loadGraph().catch(err => setStatus(err.message, true)));
controls.asOfToggle.addEventListener("change", () => {
  updateAsOfLabel();
  loadGraph().catch(err => setStatus(err.message, true));
});
let sliderTimer = null;
controls.asOfSlider.addEventListener("input", () => {
  updateAsOfLabel();
  clearTimeout(sliderTimer);
  sliderTimer = setTimeout(() => {
    if (controls.asOfToggle.checked) loadGraph().catch(err => setStatus(err.message, true));
  }, 200);
});
controls.group.addEventListener("change", () => loadTimelineBounds());
controls.kinds.forEach(input => input.addEventListener("change", render));
svg.on("click", clearSelection);
window.addEventListener("resize", () => render());

loadTimelineBounds();

loadGraph()
  .then(() => setTimeout(fitToView, 650))
  .catch(err => setStatus(err.message, true));
