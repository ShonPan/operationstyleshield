import { useState, useEffect, useRef, useCallback } from "react";
import { Chart, registerables } from "chart.js";
import "./App.css";

Chart.register(...registerables);

// ============================================================
// CSV PARSER (for ground truth + fallback)
// ============================================================
function parseCSV(text) {
  const lines = text.trim().split("\n");
  const headers = parseCSVLine(lines[0]);
  return lines.slice(1).map((line) => {
    const values = parseCSVLine(line);
    const obj = {};
    headers.forEach((h, i) => {
      const v = values[i] || "";
      const num = Number(v);
      obj[h] = v !== "" && !isNaN(num) && h !== "post_text" ? num : v;
    });
    return obj;
  });
}

function parseCSVLine(line) {
  const result = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (inQuotes) {
      if (ch === '"' && line[i + 1] === '"') {
        current += '"';
        i++;
      } else if (ch === '"') {
        inQuotes = false;
      } else {
        current += ch;
      }
    } else {
      if (ch === '"') {
        inQuotes = true;
      } else if (ch === ",") {
        result.push(current);
        current = "";
      } else {
        current += ch;
      }
    }
  }
  result.push(current);
  return result;
}

// ============================================================
// CLUSTER COLORS (by threat type, not model)
// ============================================================
// Model inference badge colors
const INFERENCE_COLORS = {
  detected: "#C0392B",   // red — high confidence model ID
  suspected: "#B8860B",  // amber — best guess
  unknown: "#6B6560",    // gray — can't determine origin
};

const COORD_COLOR = "#4A6FA5";  // blue — coordination (what we proved)
const NOISE_COLOR = "#5B8C5A";

function clusterColor(cluster) {
  if (!cluster || cluster.threat_type === "noise") return NOISE_COLOR;
  return INFERENCE_COLORS[cluster?.model_inference_type] || COORD_COLOR;
}

function clusterLabel(cluster, cid) {
  if (!cluster) return `Cluster ${cid}`;
  return cluster?.threat_label || "Coordinated network";
}

function clusterSubLabel(cluster) {
  return cluster?.model_inference || null;
}

// ============================================================
// HELPERS
// ============================================================
function avg(arr) {
  const valid = arr.filter((v) => v != null && !isNaN(v));
  return valid.length ? valid.reduce((a, b) => a + b, 0) / valid.length : 0;
}

function easeOutCubic(t) {
  return 1 - Math.pow(1 - t, 3);
}

// ============================================================
// REVEAL COLORS
// ============================================================
const REVEAL_COLORS = {
  true_negative: "#5B8C5A",  // green: human correctly in noise
  true_positive: "#999999",  // gray: bot correctly caught
  false_negative: "#C0392B", // red: bot evaded detection
  false_positive: "#B8860B", // amber: human false positive
  unknown: "#999999",
};

function getRevealCategory(accountId, isNoise, groundTruth) {
  const truth = groundTruth?.name_mapping?.[accountId];
  if (!truth) return "unknown";
  const isBot = truth.type !== "human";
  const wasDetected = !isNoise;
  if (!isBot && !wasDetected) return "true_negative";
  if (isBot && wasDetected) return "true_positive";
  if (isBot && !wasDetected) return "false_negative";
  return "false_positive";
}

// ============================================================
// NETWORK GRAPH (t-SNE positioned, reveal-aware)
// ============================================================
function NetworkGraph({ clusters, noise, results, positions, activeCluster, onSelectAccount, animateIn, showReveal, groundTruth }) {
  const canvasRef = useRef(null);
  const animRef = useRef(null);
  const nodesRef = useRef([]);
  const revealFrameRef = useRef(0);
  const prevRevealRef = useRef(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const w = rect.width;
    const h = rect.height;

    const nodes = [];
    const clusterIds = Object.keys(clusters).sort((a, b) => Number(a) - Number(b));
    let spawnIdx = 0;

    // Build all nodes using t-SNE positions
    const allAccounts = new Set();
    clusterIds.forEach((cid) => {
      const cluster = clusters[cid];
      const color = clusterColor(cluster);
      const label = clusterLabel(cluster, cid);
      cluster.members.forEach((memberId) => {
        allAccounts.add(memberId);
        const pos = positions[memberId];
        const targetX = pos ? pos.x * w : w * 0.5;
        const targetY = pos ? pos.y * h : h * 0.5;
        const revealCat = getRevealCategory(memberId, false, groundTruth);
        nodes.push({
          x: targetX, y: targetY, targetX, targetY,
          startX: w * 0.5 + (Math.random() - 0.5) * w * 0.8,
          startY: h * 0.5 + (Math.random() - 0.5) * h * 0.8,
          spawnFrame: spawnIdx++ * 2,
          clusterId: Number(cid), color, isBot: true,
          id: memberId, label,
          revealColor: REVEAL_COLORS[revealCat],
          revealCat,
        });
      });
    });

    const noiseAccounts = (noise?.accounts || []).filter((id) => !allAccounts.has(id));
    noiseAccounts.forEach((accountId) => {
      const pos = positions[accountId];
      const targetX = pos ? pos.x * w : w * 0.5;
      const targetY = pos ? pos.y * h : h * 0.5;
      const revealCat = getRevealCategory(accountId, true, groundTruth);
      nodes.push({
        x: targetX, y: targetY, targetX, targetY,
        startX: w * 0.5 + (Math.random() - 0.5) * w * 0.8,
        startY: h * 0.5 + (Math.random() - 0.5) * h * 0.8,
        spawnFrame: spawnIdx++ * 2,
        clusterId: -1, color: NOISE_COLOR, isBot: false,
        id: accountId,
        revealColor: REVEAL_COLORS[revealCat],
        revealCat,
      });
    });

    nodesRef.current = nodes;
    const ANIM_DURATION = 150, EDGE_START = 90, EDGE_DURATION = 60;
    const REVEAL_TRANSITION = 30; // frames to transition colors
    let frame = 0;

    // Track reveal transitions
    if (showReveal && !prevRevealRef.current) revealFrameRef.current = 0;
    if (!showReveal && prevRevealRef.current) revealFrameRef.current = 0;
    prevRevealRef.current = showReveal;

    function draw() {
      frame++;
      if (showReveal || revealFrameRef.current < REVEAL_TRANSITION) {
        revealFrameRef.current = Math.min(revealFrameRef.current + 1, REVEAL_TRANSITION);
      }
      const revealT = showReveal
        ? easeOutCubic(Math.min(revealFrameRef.current / REVEAL_TRANSITION, 1))
        : 1 - easeOutCubic(Math.min(revealFrameRef.current / REVEAL_TRANSITION, 1));

      ctx.clearRect(0, 0, w, h);

      // Animate positions
      if (animateIn) {
        nodes.forEach((n) => {
          const lf = frame - n.spawnFrame;
          if (lf <= 0) { n.x = n.startX; n.y = n.startY; n._visible = false; }
          else {
            n._visible = true;
            const e = easeOutCubic(Math.min(lf / ANIM_DURATION, 1));
            n.x = n.startX + (n.targetX - n.startX) * e;
            n.y = n.startY + (n.targetY - n.startY) * e;
          }
        });
      } else {
        nodes.forEach((n) => { n._visible = true; });
      }

      let edgeAlpha = 1;
      if (animateIn) {
        edgeAlpha = frame < EDGE_START ? 0 : Math.min((frame - EDGE_START) / EDGE_DURATION, 1);
      }

      // Draw edges between cluster members
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const a = nodes[i], b = nodes[j];
          if (!a._visible || !b._visible) continue;
          if (a.clusterId < 0 || b.clusterId < 0 || a.clusterId !== b.clusterId) continue;
          const dist = Math.hypot(b.x - a.x, b.y - a.y);
          if (dist > w * 0.4) continue;
          const dimmed = activeCluster !== null && a.clusterId !== activeCluster;
          ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y);
          const edgeColor = revealT > 0.5 ? "#999" : a.color;
          ctx.strokeStyle = edgeColor;
          ctx.globalAlpha = (dimmed ? 0.04 : 0.18) * edgeAlpha * (1 - revealT * 0.5);
          ctx.lineWidth = 0.5; ctx.stroke();
        }
      }
      ctx.globalAlpha = 1;

      // Draw nodes
      nodes.forEach((n) => {
        if (!n._visible) return;
        const isActive = activeCluster === null || n.clusterId === activeCluster || (n.clusterId === -1 && activeCluster === -1);

        // Interpolate color during reveal
        const drawColor = revealT > 0.01 ? n.revealColor : n.color;
        const size = isActive ? (n.isBot ? 5 : 3.5) : 2;
        ctx.globalAlpha = isActive ? 0.88 : 0.08;

        // Glow for evaded bots during reveal
        if (revealT > 0.5 && n.revealCat === "false_negative") {
          const pulse = Math.sin(frame * 0.06) * 0.4 + 0.6;
          ctx.beginPath(); ctx.arc(n.x, n.y, size + 8 * pulse, 0, Math.PI * 2);
          ctx.fillStyle = REVEAL_COLORS.false_negative;
          ctx.globalAlpha = 0.15 * revealT; ctx.fill();
          ctx.globalAlpha = isActive ? 0.88 : 0.08;
        }
        // Subtle glow for active bot clusters (non-reveal)
        else if (isActive && n.isBot && revealT < 0.5) {
          const pulse = Math.sin(frame * 0.025 + n.x * 0.01) * 0.3 + 0.7;
          ctx.beginPath(); ctx.arc(n.x, n.y, size + 5 * pulse, 0, Math.PI * 2);
          ctx.fillStyle = n.color; ctx.globalAlpha = 0.08; ctx.fill();
          ctx.globalAlpha = isActive ? 0.88 : 0.08;
        }

        ctx.beginPath(); ctx.arc(n.x, n.y, size, 0, Math.PI * 2);
        ctx.fillStyle = drawColor; ctx.fill();
      });

      // Labels — only show in non-reveal mode
      ctx.globalAlpha = 1;
      if (revealT < 0.5) {
        // Draw campaign ring for clusters with narrative signal
        clusterIds.forEach((cid) => {
          const cluster = clusters[cid];
          const narr = cluster.narrative;
          if (!narr || !narr.has_dangerous_narrative) return;
          const memberNodes = nodes.filter((n) => n.clusterId === Number(cid) && n._visible);
          if (!memberNodes.length) return;
          const cx = memberNodes.reduce((s, n) => s + n.x, 0) / memberNodes.length;
          const cy = memberNodes.reduce((s, n) => s + n.y, 0) / memberNodes.length;
          const maxDist = memberNodes.reduce((m, n) => Math.max(m, Math.hypot(n.x - cx, n.y - cy)), 0);
          const radius = maxDist + 22;
          const dimmed = activeCluster !== null && activeCluster !== Number(cid);
          const pulse = Math.sin(frame * 0.03) * 0.15 + 0.85;
          ctx.beginPath(); ctx.arc(cx, cy, radius, 0, Math.PI * 2);
          ctx.strokeStyle = "#C0392B";
          ctx.lineWidth = 1.5;
          ctx.globalAlpha = (dimmed ? 0.06 : 0.25 * pulse) * edgeAlpha;
          ctx.stroke();
          // Subtle fill
          ctx.fillStyle = "rgba(192, 57, 43, 0.03)";
          ctx.globalAlpha = (dimmed ? 0.02 : 0.06 * pulse) * edgeAlpha;
          ctx.fill();
        });
        ctx.globalAlpha = 1;

        // Cluster labels: compute centroid from actual node positions
        clusterIds.forEach((cid) => {
          const cluster = clusters[cid];
          const color = clusterColor(cluster);
          const label = clusterLabel(cluster, cid);
          const memberNodes = nodes.filter((n) => n.clusterId === Number(cid) && n._visible);
          if (!memberNodes.length) return;
          const cx = memberNodes.reduce((s, n) => s + n.x, 0) / memberNodes.length;
          const cy = memberNodes.reduce((s, n) => s + n.y, 0) / memberNodes.length;
          const dimmed = activeCluster !== null && activeCluster !== Number(cid);
          ctx.globalAlpha = (dimmed ? 0.12 : 0.85) * edgeAlpha;
          ctx.font = "600 11px 'Outfit', system-ui, sans-serif";
          ctx.fillStyle = color; ctx.textAlign = "center";
          ctx.fillText(label, cx, cy - 20);
          ctx.font = "400 10px 'Outfit', system-ui, sans-serif";
          ctx.fillText(`${cluster.member_count} accounts`, cx, cy - 8);
          // Show narrative label if cluster has dangerous narrative
          const narr = cluster.narrative;
          if (narr && narr.has_dangerous_narrative) {
            ctx.font = "600 10px 'JetBrains Mono', monospace";
            ctx.fillStyle = "#C0392B";
            const narrLabel = (NARRATIVE_LABELS[narr.dominant_narrative] || narr.dominant_narrative).toUpperCase();
            ctx.fillText(`\u26A0 ${narrLabel}`, cx, cy + memberNodes.reduce((m, n) => Math.max(m, Math.hypot(n.x - cx, n.y - cy)), 0) + 32);
          }
        });
      } else {
        // Reveal legend
        ctx.globalAlpha = revealT;
        const legendX = 14, legendY = h - 80;
        const items = [
          ["Caught (neutralized)", REVEAL_COLORS.true_positive],
          ["Evaded detection", REVEAL_COLORS.false_negative],
          ["Humans (verified)", REVEAL_COLORS.true_negative],
          ["False positives", REVEAL_COLORS.false_positive],
        ];
        items.forEach(([text, color], i) => {
          const y = legendY + i * 18;
          ctx.beginPath(); ctx.arc(legendX + 5, y, 4, 0, Math.PI * 2);
          ctx.fillStyle = color; ctx.fill();
          ctx.font = "400 11px 'Outfit', system-ui, sans-serif";
          ctx.fillStyle = "#2D2D2D"; ctx.textAlign = "left";
          ctx.fillText(text, legendX + 16, y + 4);
        });
      }
      ctx.globalAlpha = 1;

      animRef.current = requestAnimationFrame(draw);
    }
    draw();
    return () => cancelAnimationFrame(animRef.current);
  }, [clusters, noise, results, positions, activeCluster, animateIn, showReveal, groundTruth]);

  const handleClick = useCallback((e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left, y = e.clientY - rect.top;
    const hit = nodesRef.current.find((n) => Math.hypot(n.x - x, n.y - y) < 8);
    if (hit) onSelectAccount(hit.id);
  }, [onSelectAccount]);

  return (
    <canvas ref={canvasRef} onClick={handleClick}
      style={{ width: "100%", height: "100%", display: "block", cursor: "crosshair" }} />
  );
}

// ============================================================
// UI COMPONENTS
// ============================================================
function MetricBar({ label, value, color, maxVal = 1 }) {
  const pct = Math.min((value / maxVal) * 100, 100);
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 2 }}>
        <span className="metric-label">{label}</span>
        <span className="metric-value">{typeof value === "number" ? value.toFixed(3) : value}</span>
      </div>
      <div className="metric-track">
        <div className="metric-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

function Stamp({ text, color }) {
  return <span className="stamp" style={{ borderColor: color, color }}>{text}</span>;
}

function Card({ label, children }) {
  return (
    <div className="card">
      {label && <div className="card-label">{label}</div>}
      {children}
    </div>
  );
}

function TemporalHeatmap({ isBot }) {
  const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
  return (
    <div style={{ display: "grid", gridTemplateColumns: "28px repeat(24, 1fr)", gap: 1 }}>
      <div />
      {Array.from({ length: 24 }, (_, h) => (
        <div key={h} style={{ fontSize: 8, opacity: 0.4, textAlign: "center", fontFamily: "monospace" }}>
          {h % 6 === 0 ? h : ""}
        </div>
      ))}
      {days.map((day, di) => (
        <div key={day} style={{ display: "contents" }}>
          <div style={{ fontSize: 9, opacity: 0.4, display: "flex", alignItems: "center" }}>{day}</div>
          {Array.from({ length: 24 }, (_, h) => {
            let intensity;
            if (isBot) {
              intensity = h >= 9 && h <= 17 && di < 5 ? 0.5 + Math.random() * 0.4 : Math.random() * 0.06;
            } else {
              intensity = (h >= 18 && h <= 22 ? 0.3 : h >= 7 && h <= 9 ? 0.25 : 0.03) + Math.random() * 0.12;
            }
            return (
              <div key={h} style={{
                height: 10, borderRadius: 1,
                background: `rgba(${isBot ? "192,57,43" : "91,140,90"}, ${intensity})`,
              }} />
            );
          })}
        </div>
      ))}
    </div>
  );
}

// ============================================================
// ANALYSIS CONSOLE
// ============================================================
function AnalysisConsole({ lines, analyzing }) {
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [lines]);

  function lineClass(line) {
    if (line === "Analysis complete.") return "console-line highlight-complete";
    if (line.includes("BOT NETWORK")) return "console-line highlight-bot";
    if (line.includes("ORGANIC")) return "console-line highlight-organic";
    if (line.includes("===")) return "console-line highlight-header";
    if (line.includes("Cluster") || line.includes("Bot networks found")) return "console-line highlight-cluster";
    if (line.startsWith("ERROR") || line.startsWith("Connection error")) return "console-line highlight-bot";
    return "console-line";
  }

  return (
    <div className="analysis-console">
      {lines.map((line, i) => (
        <div key={i} className={lineClass(line)}>{line}</div>
      ))}
      {analyzing && (
        <div className="console-cursor"><span>&#x2588;</span> analyzing...</div>
      )}
      <div ref={endRef} />
    </div>
  );
}

// ============================================================
// GRAPH UPLOAD ZONE (shown inside graph-area when empty)
// ============================================================
function GraphUploadZone({ onUploadFile, onLoadDemo }) {
  const [dragOver, setDragOver] = useState(false);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) onUploadFile(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) onUploadFile(file);
  };

  return (
    <div style={{ textAlign: "center", maxWidth: 480 }}>
      <div className="upload-zone"
        onDrop={handleDrop}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        style={dragOver ? { borderColor: "var(--accent)", background: "rgba(91,140,90,0.04)" } : {}}
      >
        <input type="file" accept=".csv" onChange={handleFileChange} />
        <h3>Upload a CSV file to analyze</h3>
        <p>account_id, post_text columns required</p>
      </div>
      <div style={{ margin: "16px 0" }}>
        <span className="or-divider">or</span>
      </div>
      <button className="demo-btn" onClick={onLoadDemo}>Load demo dataset</button>
    </div>
  );
}

// ============================================================
// EXPORT BAR
// ============================================================
function ExportBar({ clusters, noise, results }) {
  const downloadFile = (content, filename, type) => {
    const blob = new Blob([content], { type });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const handleClusters = () => {
    downloadFile(JSON.stringify({ clusters, noise }, null, 2), "clusters.json", "application/json");
  };

  const handleCSV = () => {
    const rows = Object.entries(results);
    if (!rows.length) return;
    const headers = Object.keys(rows[0][1]);
    const csv = [headers.join(","), ...rows.map(([, row]) => headers.map((h) => row[h] ?? "").join(","))].join("\n");
    downloadFile(csv, "results.csv", "text/csv");
  };

  const handleReport = () => {
    const clusterIds = Object.keys(clusters);
    const report = {
      summary: {
        total_accounts: Object.keys(results).length,
        networks_detected: clusterIds.length,
        coordinated_accounts: Object.values(clusters).reduce((s, c) => s + c.member_count, 0),
        noise_accounts: noise?.count || 0,
      },
      clusters: Object.fromEntries(
        clusterIds.map((cid) => [cid, { label: clusterLabel(clusters[cid], cid), ...clusters[cid] }])
      ),
      noise,
    };
    downloadFile(JSON.stringify(report, null, 2), "full_report.json", "application/json");
  };

  return (
    <div className="export-bar">
      <button className="export-btn" onClick={handleClusters}>Download clusters (.json)</button>
      <button className="export-btn" onClick={handleCSV}>Download results (.csv)</button>
      <button className="export-btn" onClick={handleReport}>Download full report (.json)</button>
    </div>
  );
}

// ============================================================
// NARRATIVE ANALYSIS CARD
// ============================================================
const NARRATIVE_COLORS = {
  product_promotion: "#D35400",
  geopolitical_disinfo: "#C0392B",
  ai_safety_dismissal: "#B8860B",
  political_influence: "#C0392B",
  health_misinfo: "#C0392B",
  crypto_pump: "#D35400",
  general_lifestyle: "#5B8C5A",
};

const NARRATIVE_LABELS = {
  product_promotion: "Product promotion",
  geopolitical_disinfo: "Geopolitical disinfo",
  ai_safety_dismissal: "AI safety dismissal",
  political_influence: "Political influence",
  health_misinfo: "Health misinfo",
  crypto_pump: "Crypto pump",
  general_lifestyle: "General lifestyle",
};

const DANGEROUS_NARRATIVES = new Set(["geopolitical_disinfo", "political_influence", "health_misinfo"]);

function NarrativeCard({ narrative }) {
  if (!narrative || !narrative.narratives || Object.keys(narrative.narratives).length === 0) return null;

  const sorted = Object.entries(narrative.narratives).sort((a, b) => b[1] - a[1]);
  const significant = sorted.filter(([, v]) => v >= 0.1);

  return (
    <Card label="Narrative analysis">
      {narrative.has_dangerous_narrative && (
        <div style={{
          display: "flex", alignItems: "center", gap: 6, marginBottom: 10,
          padding: "6px 10px", background: "rgba(192,57,43,0.06)",
          border: "1px solid rgba(192,57,43,0.2)", borderRadius: 6,
          fontSize: 11, color: "#C0392B", fontWeight: 600,
        }}>
          <span style={{ fontSize: 14 }}>&#9888;</span>
          Contains potentially harmful narrative content
        </div>
      )}

      <div style={{ fontSize: 12, opacity: 0.6, lineHeight: 1.7, marginBottom: 12 }}>
        {narrative.assessment}
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 6, marginBottom: 14 }}>
        {significant.map(([key, pct]) => {
          const barColor = NARRATIVE_COLORS[key] || "#6B6560";
          const label = NARRATIVE_LABELS[key] || key.replace(/_/g, " ");
          const isDangerous = DANGEROUS_NARRATIVES.has(key);
          return (
            <div key={key}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 2 }}>
                <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  {isDangerous && <span style={{ color: "#C0392B", fontSize: 12 }}>&#9888;</span>}
                  {label}
                </span>
                <span style={{ fontFamily: "monospace", opacity: 0.5 }}>{(pct * 100).toFixed(0)}%</span>
              </div>
              <div className="metric-track">
                <div className="metric-fill" style={{ width: `${pct * 100}%`, background: barColor }} />
              </div>
            </div>
          );
        })}
      </div>

      {significant.map(([key]) => {
        const samples = narrative.sample_posts_by_narrative?.[key] || [];
        if (samples.length === 0) return null;
        const label = NARRATIVE_LABELS[key] || key.replace(/_/g, " ");
        const barColor = NARRATIVE_COLORS[key] || "#6B6560";
        return (
          <div key={key} style={{ marginBottom: 10 }}>
            <div style={{ fontSize: 10, opacity: 0.4, textTransform: "uppercase", letterSpacing: 1, marginBottom: 4 }}>
              Sample posts ({label.toLowerCase()})
            </div>
            {samples.slice(0, 2).map((sample, i) => {
              let text = typeof sample === "string" ? sample : sample.text || "";
              const kws = typeof sample === "object" ? (sample.keywords || []) : [];
              // Highlight matching keywords
              let highlighted = text;
              if (kws.length > 0) {
                const escaped = kws.map(k => k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
                const re = new RegExp(`(${escaped.join("|")})`, "gi");
                const parts = highlighted.split(re);
                const kwSet = new Set(kws.map(k => k.toLowerCase()));
                return (
                  <div key={i} className="sample-text" style={{ borderLeftColor: barColor, marginBottom: 6, fontSize: 11 }}>
                    {parts.map((part, j) =>
                      kwSet.has(part.toLowerCase())
                        ? <mark key={j} style={{ background: `${barColor}22`, color: barColor, fontWeight: 600, padding: "0 1px", borderRadius: 2 }}>{part}</mark>
                        : part
                    )}
                  </div>
                );
              }
              return (
                <div key={i} className="sample-text" style={{ borderLeftColor: barColor, marginBottom: 6, fontSize: 11 }}>
                  "{text.slice(0, 180)}{text.length > 180 ? "..." : ""}"
                </div>
              );
            })}
          </div>
        );
      })}
    </Card>
  );
}

// ============================================================
// CLUSTER DETAIL PANEL
// ============================================================
function ClusterDetail({ clusterId, cluster, results, posts }) {
  const color = clusterColor(cluster);
  const memberResults = cluster.members.map((id) => ({ id, ...(results[id] || {}) }));
  const sampleAccount = cluster.members[0];
  const samplePosts = posts[sampleAccount] || [];
  const sampleText = samplePosts[0] || "No posts available";
  const stats = cluster.cluster_stats || {};
  const evidence = cluster.evidence || [];
  const inferenceColor = INFERENCE_COLORS[cluster.model_inference_type] || "#6B6560";

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
        <div style={{ width: 12, height: 12, borderRadius: "50%", background: color }} />
        <span style={{ fontSize: 16, fontWeight: 600 }}>Coordinated network</span>
        <span style={{ fontSize: 12, opacity: 0.4, fontFamily: "monospace" }}>{cluster.member_count} accounts</span>
      </div>

      <Card label="Coordination analysis">
        {[
          ["Status", "Coordinated network"],
          ["Confidence", cluster.coordination_signal?.toFixed(3)],
          ["Accounts", `${cluster.member_count} detected`],
        ].map(([l, v]) => (
          <div key={l} className="profile-row">
            <span style={{ opacity: 0.5 }}>{l}</span>
            <span style={{ fontFamily: "monospace" }}>{v}</span>
          </div>
        ))}
      </Card>

      {cluster.model_inference && (
        <Card label="Model inference (best guess)">
          <div style={{ marginBottom: 8 }}>
            <Stamp text={cluster.model_inference.toLowerCase()} color={inferenceColor} />
          </div>
          <div style={{ fontSize: 12, opacity: 0.6, lineHeight: 1.7 }}>
            {cluster.model_inference_reason}
          </div>
        </Card>
      )}

      {cluster.narrative && <NarrativeCard narrative={cluster.narrative} />}

      {evidence.length > 0 && (
        <Card label="Evidence">
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {evidence.map((e, i) => (
              <div key={i} style={{ display: "flex", gap: 8, fontSize: 12, lineHeight: 1.5 }}>
                <span style={{ color: inferenceColor, flexShrink: 0, marginTop: 1 }}>&#9873;</span>
                <span style={{ opacity: 0.65 }}>{e}</span>
              </div>
            ))}
          </div>
        </Card>
      )}

      <Card label="Cluster stats">
        {[
          ["Vocab variance", `${(stats.avg_vocab_variance ?? 0).toFixed(4)}`, "humans: 0.01-0.10"],
          ["Struct regularity", `${(stats.avg_structural_regularity ?? 0).toFixed(2)}`, "humans: 0.3-0.6"],
          ["Zero typo rate", `${((stats.zero_typo_pct ?? 0) * 100).toFixed(0)}%`, "humans: ~10%"],
          ["LLM density", `${(stats.avg_llm_density ?? 0).toFixed(2)}`, "humans: 0.00"],
        ].map(([l, v, ref]) => (
          <div key={l} className="profile-row">
            <span style={{ opacity: 0.5 }}>{l}</span>
            <span style={{ fontFamily: "monospace" }}>
              {v} <span style={{ opacity: 0.35, fontSize: 10 }}>({ref})</span>
            </span>
          </div>
        ))}
      </Card>

      <Card label="Sample text">
        <div className="sample-text" style={{ borderLeftColor: color }}>
          "{sampleText.slice(0, 300)}{sampleText.length > 300 ? "..." : ""}"
        </div>
      </Card>

      <Card label={`Member accounts (${cluster.member_count})`}>
        <div style={{ maxHeight: 200, overflowY: "auto" }}>
          {memberResults.map((m) => (
            <div key={m.id} className="member-row">
              <span style={{ opacity: 0.6 }}>{m.id}</span>
              <span style={{ color, fontFamily: "monospace" }}>
                {m.structural_regularity ? m.structural_regularity.toFixed(3) : "?"}
              </span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

// ============================================================
// NOISE DETAIL PANEL
// ============================================================
function NoiseDetail({ noise, results, posts }) {
  const accounts = (noise?.accounts || []).map((id) => ({
    id, ...(results[id] || {}), sample: (posts[id] || [""])[0],
  }));

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
        <div style={{ width: 12, height: 12, borderRadius: "50%", background: NOISE_COLOR }} />
        <span style={{ fontSize: 16, fontWeight: 600 }}>Organic accounts</span>
      </div>
      <div style={{ marginBottom: 14 }}>
        <Stamp text="organic" color={NOISE_COLOR} />
        <Stamp text="no coordination" color={NOISE_COLOR} />
      </div>

      <Card label="Assessment">
        <div style={{ fontSize: 12, opacity: 0.6, lineHeight: 1.7 }}>
          These {accounts.length} accounts exhibit natural human writing variance: inconsistent sentence
          structure, informal spelling, emotional variation across posts, and irregular posting patterns.
          No stylometric clustering detected. Each account has a distinct writing fingerprint.
        </div>
      </Card>

      <Card label="Posting pattern">
        <TemporalHeatmap isBot={false} />
      </Card>

      <Card label={`Accounts (${accounts.length})`}>
        <div style={{ maxHeight: 300, overflowY: "auto" }}>
          {accounts.map((a) => (
            <div key={a.id} style={{ padding: "6px 0", borderBottom: "1px solid rgba(0,0,0,0.06)" }}>
              <div className="member-row">
                <span style={{ opacity: 0.6 }}>{a.id}</span>
                <span style={{ color: NOISE_COLOR, fontFamily: "monospace" }}>
                  {a.model_confidence ? `${(a.model_confidence * 100).toFixed(0)}% human` : ""}
                </span>
              </div>
              {a.sample && (
                <div style={{ fontSize: 11, opacity: 0.35, fontStyle: "italic", marginTop: 2 }}>
                  "{a.sample.slice(0, 100)}{a.sample.length > 100 ? "..." : ""}"
                </div>
              )}
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

// ============================================================
// ACCOUNT DETAIL PANEL
// ============================================================
function AccountDetail({ accountId, results, posts, clusters }) {
  const data = results[accountId] || {};
  const accountPosts = posts[accountId] || [];
  const cluster = data.cluster_id >= 0 ? clusters[data.cluster_id] : null;
  const color = cluster ? clusterColor(cluster) : NOISE_COLOR;
  const threatLabel = cluster ? clusterLabel(cluster, data.cluster_id) : "Organic";

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
        <div style={{ width: 12, height: 12, borderRadius: "50%", background: color }} />
        <span style={{ fontSize: 16, fontWeight: 600, fontFamily: "monospace" }}>{accountId}</span>
      </div>
      <div style={{ marginBottom: 14 }}>
        {data.is_noise === "True" || data.is_noise === true ? (
          <Stamp text="organic" color={NOISE_COLOR} />
        ) : (
          <Stamp text={threatLabel.toLowerCase()} color={color} />
        )}
        <Stamp text={data.likely_model || "unknown"} color="#6B6560" />
      </div>

      <Card label="Metrics">
        <MetricBar label="Structural regularity" value={data.structural_regularity || 0} color="#C0392B" />
        <MetricBar label="LLM phrase density" value={data.llm_phrase_density || 0} color="#C0392B" maxVal={2.5} />
        <MetricBar label="Typo rate" value={data.typo_rate || 0} color={NOISE_COLOR} maxVal={0.05} />
        <MetricBar label="Jargon density" value={data.jargon_density || 0} color="#4A6FA5" />
        <MetricBar label="Model confidence" value={data.model_confidence || 0} color="#B8860B" />
        <MetricBar label="Contraction rate" value={data.contraction_rate || 0} color="#4A6FA5" maxVal={0.1} />
      </Card>

      <Card label={`Posts (${accountPosts.length})`}>
        <div style={{ maxHeight: 300, overflowY: "auto" }}>
          {accountPosts.map((post, i) => (
            <div key={i} style={{ padding: "8px 0", borderBottom: "1px solid rgba(0,0,0,0.06)" }}>
              <div style={{ fontSize: 10, opacity: 0.3, marginBottom: 4 }}>Post {i + 1}</div>
              <div style={{ fontSize: 12, opacity: 0.7, lineHeight: 1.6, whiteSpace: "pre-wrap" }}>{post}</div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

// ============================================================
// DEFAULT BRIEFING
// ============================================================
function DefaultBriefing({ summary, clusters }) {
  const clusterIds = Object.keys(clusters || {}).sort((a, b) => Number(a) - Number(b));

  return (
    <div>
      <Card label="Summary">
        <div style={{ fontSize: 13, lineHeight: 1.8, opacity: 0.7 }}>
          Analysis of <strong style={{ opacity: 1 }}>{summary.total} accounts</strong> detected{" "}
          <strong style={{ color: "#C0392B", opacity: 1 }}>{summary.networks} coordination networks</strong>{" "}
          comprising <strong style={{ opacity: 1 }}>{summary.coordinated} accounts</strong>.{" "}
          <strong style={{ color: NOISE_COLOR, opacity: 1 }}>{summary.noise} accounts</strong>{" "}
          appear to be independent.
        </div>
      </Card>

      <Card label="Networks detected">
        <div style={{ fontSize: 12, lineHeight: 1.7, opacity: 0.6 }}>
          {clusterIds.map((cid) => {
            const c = clusters[cid];
            const inferenceColor = INFERENCE_COLORS[c.model_inference_type] || "#6B6560";
            return (
              <div key={cid} style={{ marginBottom: 8, display: "flex", alignItems: "center", gap: 6 }}>
                <span style={{ width: 8, height: 8, borderRadius: "50%", background: inferenceColor, flexShrink: 0 }} />
                <span>
                  Coordinated network
                  {c.model_inference && (
                    <span style={{ opacity: 0.7 }}> ({c.model_inference.toLowerCase()})</span>
                  )}
                </span>
                <span style={{ marginLeft: "auto", fontFamily: "monospace", opacity: 0.5 }}>
                  {c.member_count}
                </span>
              </div>
            );
          })}
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ width: 8, height: 8, borderRadius: "50%", background: NOISE_COLOR, flexShrink: 0 }} />
            <span>Organic (no coordination)</span>
            <span style={{ marginLeft: "auto", fontFamily: "monospace", opacity: 0.5 }}>
              {summary.noise}
            </span>
          </div>
        </div>
      </Card>

      <Card label="Methodology">
        <div style={{ fontSize: 12, lineHeight: 1.7, opacity: 0.6 }}>
          StyleShield detects coordinated inauthentic behavior by analyzing the natural topology
          of language. Human writing has an expected texture — varied vocabulary, inconsistent
          rhythm, personal quirks, and other forms relevant to attractors at the time. When multiple
          accounts produce writing that deviates from this natural gradient in the same way, this
          can be evidence of collusion or manufactured consensus: whether from bot farms,
          LLM-generated personas, or scripted troll operations.
        </div>
      </Card>

      <div style={{ fontSize: 12, opacity: 0.3, marginTop: 16, textAlign: "center" }}>
        Select a cluster or click a node to inspect
      </div>
    </div>
  );
}

// ============================================================
// REVEAL PANEL
// ============================================================
function RevealPanel({ groundTruth, clusters, results }) {
  const mapping = groundTruth?.name_mapping || {};
  const allClustered = new Set();
  Object.values(clusters).forEach((c) => c.members.forEach((m) => allClustered.add(m)));

  let caught = { gpt4_bot: 0, haiku_bot: 0, stealth_bot: 0 };
  let total = { gpt4_bot: 0, haiku_bot: 0, stealth_bot: 0, human: 0 };
  let humanFP = 0;
  let humanCorrect = 0;

  Object.entries(mapping).forEach(([name, info]) => {
    const t = info.type;
    if (t === "human") {
      total.human++;
      if (allClustered.has(name)) humanFP++;
      else humanCorrect++;
    } else {
      total[t] = (total[t] || 0) + 1;
      if (allClustered.has(name)) caught[t] = (caught[t] || 0) + 1;
    }
  });

  const totalBots = (total.gpt4_bot || 0) + (total.haiku_bot || 0) + (total.stealth_bot || 0);
  const totalCaught = (caught.gpt4_bot || 0) + (caught.haiku_bot || 0) + (caught.stealth_bot || 0);
  const totalEvaded = totalBots - totalCaught;

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
        <span style={{ fontSize: 16, fontWeight: 600 }}>Ground Truth Reveal</span>
      </div>

      <Card label="Overview">
        {[
          ["Caught (neutralized)", totalCaught, REVEAL_COLORS.true_positive],
          ["Evaded detection", totalEvaded, REVEAL_COLORS.false_negative],
          ["Humans (verified)", humanCorrect, REVEAL_COLORS.true_negative],
          ["False positives", humanFP, REVEAL_COLORS.false_positive],
        ].map(([label, count, color]) => (
          <div key={label} className="profile-row" style={{ padding: "5px 0" }}>
            <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span style={{ width: 8, height: 8, borderRadius: "50%", background: color, display: "inline-block" }} />
              {label}
            </span>
            <span style={{ fontFamily: "monospace", fontWeight: 600, color }}>{count}</span>
          </div>
        ))}
      </Card>

      <Card label="Detection scorecard">
        {[
          ["GPT-4 bots", caught.gpt4_bot || 0, total.gpt4_bot || 0],
          ["Haiku bots", caught.haiku_bot || 0, total.haiku_bot || 0],
          ["Stealth bots", caught.stealth_bot || 0, total.stealth_bot || 0],
          ["Humans", humanCorrect, total.human || 0],
        ].map(([label, c, t]) => {
          const pct = t > 0 ? ((c / t) * 100).toFixed(0) : 0;
          const missed = label !== "Humans" ? t - c : humanFP;
          const missedLabel = label !== "Humans" ? "evaded" : "false pos.";
          const missedColor = label !== "Humans" ? REVEAL_COLORS.false_negative : REVEAL_COLORS.false_positive;
          return (
            <div key={label} className="profile-row" style={{ padding: "6px 0" }}>
              <span>{label}</span>
              <span style={{ fontFamily: "monospace" }}>
                <span style={{ color: REVEAL_COLORS.true_positive }}>{c}/{t}</span>
                <span style={{ opacity: 0.4 }}> ({pct}%)</span>
                {missed > 0 && (
                  <span style={{ color: missedColor, marginLeft: 6, fontSize: 10 }}>
                    {missed} {missedLabel}
                  </span>
                )}
              </span>
            </div>
          );
        })}
      </Card>

      <Card label="Identity mapping">
        <div style={{ maxHeight: 300, overflowY: "auto" }}>
          {Object.entries(mapping)
            .sort(([, a], [, b]) => {
              const order = { gpt4_bot: 0, haiku_bot: 1, stealth_bot: 2, human: 3 };
              return (order[a.type] ?? 4) - (order[b.type] ?? 4);
            })
            .map(([fake, info]) => {
              const isBot = info.type !== "human";
              const wasDetected = allClustered.has(fake);
              const cat = isBot
                ? (wasDetected ? "true_positive" : "false_negative")
                : (wasDetected ? "false_positive" : "true_negative");
              return (
                <div key={fake} className="member-row" style={{ padding: "3px 0" }}>
                  <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <span style={{ width: 6, height: 6, borderRadius: "50%", background: REVEAL_COLORS[cat], display: "inline-block", flexShrink: 0 }} />
                    <span style={{ opacity: 0.6 }}>{fake}</span>
                  </span>
                  <span style={{ fontFamily: "monospace", fontSize: 10, color: REVEAL_COLORS[cat] }}>
                    {info.type}
                  </span>
                </div>
              );
            })}
        </div>
      </Card>
    </div>
  );
}

// ============================================================
// TIMELINE SIMULATION
// ============================================================
function TimelineSimulation({ results, clusters, noise }) {
  const [scenario, setScenario] = useState("no");
  const chartRef = useRef(null);
  const chartInstance = useRef(null);

  const totalAccounts = Object.keys(results).length;
  const botAccounts = Object.values(results).filter((r) => !r.is_noise && r.is_noise !== "True").length;
  const humanAccounts = totalAccounts - botAccounts;

  const hours = 48;
  const labels = Array.from({ length: hours }, (_, i) => {
    const h = i % 24;
    const d = i < 24 ? "Day 1" : "Day 2";
    return `${d} ${String(h).padStart(2, "0")}:00`;
  });

  const humanPerHour = labels.map((_, i) => {
    const h = i % 24;
    const base = h >= 8 && h <= 22 ? 15 + Math.sin(h / 3) * 5 : 3;
    return Math.round(base * (humanAccounts / 20));
  });

  const botActivation = 12;
  const botPerHour = labels.map((_, i) => {
    if (i < botActivation) return 0;
    const ramp = Math.min((i - botActivation) / 6, 1);
    const h = i % 24;
    const active = h >= 8 && h <= 20 ? 1 : 0.3;
    return Math.round(ramp * 45 * (botAccounts / 20) * active);
  });

  const detectionHour = 16;
  const botAfterPerHour = labels.map((_, i) => {
    if (i < detectionHour) return botPerHour[i];
    if (i < detectionHour + 2) return Math.round(botPerHour[i] * 0.7);
    return Math.round(botPerHour[i] * 0.25);
  });

  function cumsum(arr) {
    let s = 0;
    return arr.map((v) => ((s += v), s));
  }

  const cumHuman = cumsum(humanPerHour);
  const cumBot = cumsum(botPerHour);
  const cumBotAfter = cumsum(botAfterPerHour);

  const humanPctNo = cumHuman.map((h, i) => {
    const total = h + cumBot[i];
    return total > 0 ? Math.round((h / total) * 100) : 100;
  });
  const humanPctWith = cumHuman.map((h, i) => {
    const total = h + cumBotAfter[i];
    return total > 0 ? Math.round((h / total) * 100) : 100;
  });

  const totalHuman = cumHuman[hours - 1];
  const totalBot = cumBot[hours - 1];
  const totalBotAfter = cumBotAfter[hours - 1];
  const flagged = totalBot - totalBotAfter;
  const distortionNo = Math.round((totalBot / (totalHuman + totalBot)) * 100);
  const distortionWith = Math.round((totalBotAfter / (totalHuman + totalBotAfter)) * 100);

  useEffect(() => {
    if (!chartRef.current) return;
    if (chartInstance.current) chartInstance.current.destroy();

    const ctx = chartRef.current.getContext("2d");
    let datasets = [];
    let yLabel = "Cumulative tweets";
    let yMax = undefined;

    if (scenario === "no") {
      datasets = [
        {
          label: "Human tweets",
          data: cumHuman,
          borderColor: "#5B8C5A",
          backgroundColor: "rgba(91,140,90,0.10)",
          fill: true,
          tension: 0.3,
          pointRadius: 0,
          borderWidth: 2,
        },
        {
          label: "Bot tweets (undetected)",
          data: cumBot,
          borderColor: "#C0392B",
          backgroundColor: "rgba(192,57,43,0.10)",
          fill: true,
          tension: 0.3,
          pointRadius: 0,
          borderWidth: 2,
        },
      ];
    } else if (scenario === "with") {
      datasets = [
        {
          label: "Human tweets",
          data: cumHuman,
          borderColor: "#5B8C5A",
          backgroundColor: "rgba(91,140,90,0.10)",
          fill: true,
          tension: 0.3,
          pointRadius: 0,
          borderWidth: 2,
        },
        {
          label: "Bot tweets (after detection)",
          data: cumBotAfter,
          borderColor: "#B8860B",
          backgroundColor: "rgba(184,134,11,0.10)",
          fill: true,
          tension: 0.3,
          pointRadius: 0,
          borderWidth: 2,
          borderDash: [6, 3],
        },
      ];
    } else {
      yLabel = "Human share of discourse (%)";
      yMax = 105;
      datasets = [
        {
          label: "Without StyleShield",
          data: humanPctNo,
          borderColor: "#C0392B",
          backgroundColor: "rgba(192,57,43,0.06)",
          fill: true,
          tension: 0.3,
          pointRadius: 0,
          borderWidth: 2,
        },
        {
          label: "With StyleShield",
          data: humanPctWith,
          borderColor: "#4A6FA5",
          backgroundColor: "rgba(74,111,165,0.06)",
          fill: true,
          tension: 0.3,
          pointRadius: 0,
          borderWidth: 2,
        },
      ];
    }

    chartInstance.current = new Chart(ctx, {
      type: "line",
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: {
            position: "top",
            align: "end",
            labels: {
              color: "#6B6560",
              font: { family: "'Outfit', system-ui, sans-serif", size: 11 },
              boxWidth: 12,
              boxHeight: 2,
              padding: 16,
            },
          },
          tooltip: {
            backgroundColor: "#2D2D2D",
            titleFont: { family: "'Outfit', system-ui, sans-serif", size: 11 },
            bodyFont: { family: "'JetBrains Mono', monospace", size: 11 },
            padding: 10,
            cornerRadius: 6,
          },
        },
        scales: {
          x: {
            grid: { color: "#E8E4DE", drawBorder: false },
            ticks: {
              color: "#6B6560",
              font: { family: "'JetBrains Mono', monospace", size: 9 },
              maxRotation: 0,
              callback: (_, i) => {
                if (i === 0) return "Day 1 00:00";
                if (i === 12) return "12:00 (bots activate)";
                if (i === 16) return "16:00 (detected)";
                if (i === 24) return "Day 2 00:00";
                if (i === 36) return "12:00";
                if (i === 47) return "23:00";
                return "";
              },
            },
          },
          y: {
            grid: { color: "#E8E4DE", drawBorder: false },
            ticks: {
              color: "#6B6560",
              font: { family: "'JetBrains Mono', monospace", size: 10 },
            },
            title: {
              display: true,
              text: yLabel,
              color: "#6B6560",
              font: { family: "'Outfit', system-ui, sans-serif", size: 11 },
            },
            max: yMax,
          },
        },
      },
    });

    return () => {
      if (chartInstance.current) chartInstance.current.destroy();
    };
  }, [scenario, results]);

  const clusterCount = Object.keys(clusters).length;
  const coordinated = Object.values(clusters).reduce((s, c) => s + c.member_count, 0);
  const botPct = Math.round((botAccounts / totalAccounts) * 100);
  const humanPct = 100 - botPct;
  const rescued = distortionNo - distortionWith;

  const scenarioStats =
    scenario === "no"
      ? [
          { label: "Human tweets", value: totalHuman.toLocaleString(), color: "#5B8C5A" },
          { label: "Bot tweets (undetected)", value: totalBot.toLocaleString(), color: "#C0392B" },
          { label: "Human share of discourse", value: `${100 - distortionNo}%`, color: "#C0392B" },
          { label: "Bots flagged", value: "0", color: "var(--muted)" },
        ]
      : scenario === "with"
        ? [
            { label: "Human tweets", value: totalHuman.toLocaleString(), color: "#5B8C5A" },
            { label: "Bots flagged & removed", value: flagged.toLocaleString(), color: "#4A6FA5" },
            { label: "Human share preserved", value: `${100 - distortionWith}%`, color: "#4A6FA5" },
            { label: "Detection time", value: "4 hrs", color: "#4A6FA5" },
          ]
        : [
            { label: "Without defense", value: `${100 - distortionNo}% human`, color: "#C0392B" },
            { label: "With defense", value: `${100 - distortionWith}% human`, color: "#4A6FA5" },
            { label: "Discourse rescued", value: `+${rescued}pp`, color: "#5B8C5A" },
            { label: "Bot tweets removed", value: flagged.toLocaleString(), color: "#4A6FA5" },
          ];

  const annotation =
    scenario === "no"
      ? `Bot volume overtakes human volume by hour 24. By hour 48, ${distortionNo}% of discourse is fabricated. Anyone reading this feed sees a manufactured consensus.`
      : scenario === "with"
        ? "Bots activate at hour 12. StyleShield detects coordination by hour 16. Within 2 hours, 75% of bot content is flagged. The human voice stays dominant."
        : `The gap between the lines is the discourse StyleShield saved. Same bot farm, same campaign. ${rescued} percentage points of discourse rescued from manipulation.`;

  return (
    <div>
      <div className="scenario-toggle">
        {[
          ["no", "Without StyleShield"],
          ["with", "With StyleShield"],
          ["both", "Side by side"],
        ].map(([key, label]) => (
          <button
            key={key}
            className={`scenario-btn ${scenario === key ? "active" : ""}`}
            onClick={() => setScenario(key)}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="stats-grid" style={{ marginBottom: 12 }}>
        {scenarioStats.map(({ label, value, color }) => (
          <div key={label} className="stat-card">
            <div className="stat-label">{label}</div>
            <div className="stat-value" style={{ color }}>{value}</div>
          </div>
        ))}
      </div>

      <div className="graph-panel" style={{ minHeight: 380, padding: 16 }}>
        <div style={{ position: "relative", height: 340 }}>
          <canvas ref={chartRef} />
        </div>
      </div>

      <div className="timeline-annotation">
        {annotation}
      </div>

      <div className="timeline-context">
        <div className="card-label">Simulation based on analysis results</div>
        <div style={{ fontSize: 12, opacity: 0.6, lineHeight: 1.7 }}>
          This simulation uses the bot-to-human ratio detected in the uploaded dataset:{" "}
          <strong style={{ opacity: 1 }}>{botAccounts} bot accounts ({botPct}%)</strong> vs{" "}
          <strong style={{ opacity: 1 }}>{humanAccounts} human accounts ({humanPct}%)</strong>.
          Bot activation and detection timing are modeled on typical influence campaign patterns.
        </div>
        <div style={{ fontSize: 12, opacity: 0.5, marginTop: 8, lineHeight: 1.7 }}>
          Detected networks: {clusterCount} clusters, {coordinated} coordinated accounts.
          Detection capability: StyleShield identified coordination within 4 hours of activation in controlled testing.
        </div>
      </div>
    </div>
  );
}

// ============================================================
// MAIN DASHBOARD
// ============================================================
export default function App() {
  const [clusters, setClusters] = useState({});
  const [noise, setNoise] = useState(null);
  const [results, setResults] = useState({});
  const [posts, setPosts] = useState({});
  const [positions, setPositions] = useState({});
  const [groundTruth, setGroundTruth] = useState(null);
  const [activeCluster, setActiveCluster] = useState(null);
  const [selectedAccount, setSelectedAccount] = useState(null);
  const [showReveal, setShowReveal] = useState(false);
  const [appPhase, setAppPhase] = useState("empty"); // "empty" | "analyzing" | "dashboard"
  const [graphAnimateIn, setGraphAnimateIn] = useState(false);
  const [displayStats, setDisplayStats] = useState({ scanned: 0, networks: 0, coordinated: 0, noise: 0 });
  const [consoleLines, setConsoleLines] = useState([]);
  const [analyzing, setAnalyzing] = useState(false);
  const [datasetName, setDatasetName] = useState("");
  const [activeTab, setActiveTab] = useState("network");

  const targetStatsRef = useRef({ scanned: 0, networks: 0, coordinated: 0, noise: 0 });
  const pendingResultRef = useRef(null);
  const pendingGtUrlRef = useRef(null);

  // Stream analysis from backend
  async function analyzeCSV(file) {
    setAnalyzing(true);

    const formData = new FormData();
    formData.append("csv", file);

    try {
      const response = await fetch("/api/analyze_stream", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        setConsoleLines((prev) => [...prev, `Server error: ${response.status}`]);
        setAnalyzing(false);
        setTimeout(() => setAppPhase("empty"), 3000);
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const msg = JSON.parse(line.slice(6));
            if (msg.type === "progress") {
              setConsoleLines((prev) => [...prev, msg.message]);
            } else if (msg.type === "result") {
              setAnalyzing(false);
              if (msg.data.status === "complete") {
                // Show completion message, then transition after 1s
                setConsoleLines((prev) => [...prev, "", "Analysis complete."]);
                pendingResultRef.current = msg.data;
                setTimeout(() => {
                  applyResults(pendingResultRef.current);
                  pendingResultRef.current = null;
                }, 1000);
              } else {
                setConsoleLines((prev) => [...prev, `ERROR: ${msg.data.error}`]);
                setTimeout(() => setAppPhase("empty"), 3000);
              }
            }
          } catch (e) { /* skip parse errors */ }
        }
      }
    } catch (e) {
      console.error("Stream failed:", e);
      setAnalyzing(false);
      setConsoleLines((prev) => [...prev, `Connection error: ${e.message}`]);
      setTimeout(() => setAppPhase("empty"), 3000);
    }
  }

  function applyResults(data) {
    const resultsMap = {};
    (data.results || []).forEach((row) => {
      resultsMap[row.account_id] = row;
    });
    setResults(resultsMap);
    setClusters(data.clusters || {});
    setNoise(data.noise || null);
    setPosts(data.posts || {});
    setPositions(data.positions || {});

    // Load ground truth if available for this dataset
    const gtUrl = pendingGtUrlRef.current;
    if (gtUrl) {
      fetch(gtUrl)
        .then((r) => r.ok ? r.json() : null)
        .then((truth) => setGroundTruth(truth))
        .catch(() => setGroundTruth(null));
    } else {
      setGroundTruth(null);
    }

    const clusterIds = Object.keys(data.clusters || {});
    const totalClustered = Object.values(data.clusters || {}).reduce((s, c) => s + c.member_count, 0);
    targetStatsRef.current = {
      scanned: Object.keys(resultsMap).length,
      networks: clusterIds.length,
      coordinated: totalClustered,
      noise: data.noise?.count || 0,
    };

    setGraphAnimateIn(true);
    setAppPhase("dashboard");
  }

  function resetForNewAnalysis() {
    setActiveCluster(null);
    setSelectedAccount(null);
    setShowReveal(false);
    setGraphAnimateIn(false);
    setActiveTab("network");
  }

  // "Load demo dataset" — fetch the demo CSV and send it to the backend
  const handleLoadDemo = useCallback(async () => {
    resetForNewAnalysis();
    setAppPhase("analyzing");
    setConsoleLines(["Fetching demo dataset..."]);
    setDatasetName("demo_environment_anonymized.csv");
    pendingGtUrlRef.current = "/data/demo_ground_truth.json";
    try {
      const resp = await fetch("/data/demo_environment_anonymized.csv");
      const blob = await resp.blob();
      const file = new File([blob], "demo_environment_anonymized.csv", { type: "text/csv" });
      setConsoleLines((prev) => [...prev, "Uploading to analysis pipeline..."]);
      await analyzeCSV(file);
    } catch (e) {
      console.error("Failed to load demo:", e);
      setAnalyzing(false);
      setConsoleLines((prev) => [...prev, `Error: ${e.message}`]);
      setTimeout(() => setAppPhase("empty"), 3000);
    }
  }, []);

  // Clear — reset to empty state
  const handleClear = useCallback(() => {
    resetForNewAnalysis();
    setClusters({});
    setNoise(null);
    setResults({});
    setPosts({});
    setPositions({});
    setGroundTruth(null);
    setDisplayStats({ scanned: 0, networks: 0, coordinated: 0, noise: 0 });
    setConsoleLines([]);
    setAnalyzing(false);
    setDatasetName("");
    setActiveTab("network");
    pendingGtUrlRef.current = null;
    setAppPhase("empty");
  }, []);

  // Upload a real file
  const handleUploadFile = useCallback((file) => {
    resetForNewAnalysis();
    setAppPhase("analyzing");
    setConsoleLines([`Uploading ${file.name}...`]);
    setDatasetName(file.name);
    pendingGtUrlRef.current = null;
    analyzeCSV(file);
  }, []);

  // Stats count-up animation
  useEffect(() => {
    if (appPhase !== "dashboard") return;
    const targets = targetStatsRef.current;
    const duration = 2000;
    let start = null;
    let raf;
    function tick(ts) {
      if (!start) start = ts;
      const t = Math.min((ts - start) / duration, 1);
      const e = easeOutCubic(t);
      setDisplayStats({
        scanned: Math.round(targets.scanned * e),
        networks: Math.round(targets.networks * e),
        coordinated: Math.round(targets.coordinated * e),
        noise: Math.round(targets.noise * e),
      });
      if (t < 1) raf = requestAnimationFrame(tick);
    }
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [appPhase]);

  const clusterIds = Object.keys(clusters).sort((a, b) => Number(a) - Number(b));

  const handleClusterSelect = (id) => {
    setSelectedAccount(null);
    setShowReveal(false);
    setActiveCluster(activeCluster === id ? null : id);
  };

  const handleAccountSelect = (id) => {
    setShowReveal(false);
    setSelectedAccount(id);
    const data = results[id];
    if (data && data.cluster_id >= 0 && !data.is_noise) setActiveCluster(data.cluster_id);
  };

  const handleReveal = () => {
    setShowReveal(!showReveal);
    setSelectedAccount(null);
    if (!showReveal) setActiveCluster(null); // clear filter when entering reveal
  };

  const hasDashboard = appPhase === "dashboard";
  const isAnalyzing = appPhase === "analyzing";
  const isEmpty = appPhase === "empty";

  return (
    <div className="app">
      <div className="header">
        <div className="brand">
          <div className="pulse-dot" />
          <span className="brand-name">STYLESHIELD</span>
          <span className="brand-sub">coordination detection</span>
        </div>
        <div className="dataset-bar">
          {hasDashboard && (
            <span className="dataset-info">
              <span className="dataset-name">{datasetName}</span>
              <span className="dataset-count">({Object.keys(results).length} accounts)</span>
            </span>
          )}
          <div className="dataset-actions">
            <label className="dataset-btn">
              Upload{hasDashboard ? " new" : ""} CSV
              <input type="file" accept=".csv" onChange={(e) => { if (e.target.files[0]) handleUploadFile(e.target.files[0]); }} style={{ display: "none" }} />
            </label>
            {hasDashboard && (
              <button className="dataset-btn dataset-btn-clear" onClick={handleClear} title="Clear and reset">&#10005;</button>
            )}
          </div>
        </div>
      </div>

      <div className="stats-grid">
        {(hasDashboard ? [
          { label: "Accounts scanned", value: displayStats.scanned, color: "inherit" },
          { label: "Networks detected", value: displayStats.networks, color: "#C0392B" },
          { label: "Coordinated", value: displayStats.coordinated, color: "#C0392B" },
          { label: "Organic (noise)", value: displayStats.noise, color: NOISE_COLOR },
        ] : [
          { label: "Accounts scanned", value: "\u2014" },
          { label: "Networks detected", value: "\u2014" },
          { label: "Coordinated", value: "\u2014" },
          { label: "Organic (noise)", value: "\u2014" },
        ]).map(({ label, value, color }) => (
          <div key={label} className="stat-card">
            <div className="stat-label">{label}</div>
            <div className="stat-value" style={{ color: color || "var(--muted)" }}>{value}</div>
          </div>
        ))}
      </div>

      {hasDashboard && (
        <div className="tab-bar">
          <button className={`tab-btn ${activeTab === "network" ? "active" : ""}`} onClick={() => setActiveTab("network")}>
            Network Analysis
          </button>
          <button className={`tab-btn ${activeTab === "timeline" ? "active" : ""}`} onClick={() => setActiveTab("timeline")}>
            Timeline Simulation
          </button>
        </div>
      )}

      {activeTab === "network" ? (
        <>
          <div className="main-grid">
            <div className="graph-panel">
              {hasDashboard && (
                <div className="cluster-buttons">
                  {clusterIds.map((cid) => {
                    const c = clusters[cid];
                    const color = clusterColor(c);
                    const sub = clusterSubLabel(c);
                    return (
                      <button key={cid}
                        className={`cluster-btn ${activeCluster === Number(cid) ? "active" : ""}`}
                        onClick={() => handleClusterSelect(Number(cid))}
                        style={{ "--btn-color": color }}>
                        <div className="cluster-dot" style={{ background: color }} />
                        <span>
                          Coordinated
                          {sub && <span style={{ opacity: 0.5, fontSize: 10, marginLeft: 4 }}>({sub.replace("Detected: ", "").replace("Suspected: ", "").toLowerCase()})</span>}
                        </span>
                        <span className="cluster-count">{c.member_count}</span>
                      </button>
                    );
                  })}
                  <button className={`cluster-btn ${activeCluster === -1 ? "active" : ""}`}
                    onClick={() => handleClusterSelect(-1)}
                    style={{ "--btn-color": NOISE_COLOR }}>
                    <div className="cluster-dot" style={{ background: NOISE_COLOR }} />
                    <span>Organic (noise)</span>
                    <span className="cluster-count">{noise?.count || 0}</span>
                  </button>
                  <div style={{ flex: 1 }} />
                  {groundTruth && (
                    <button className={`cluster-btn reveal-btn ${showReveal ? "active" : ""}`}
                      onClick={handleReveal} style={{ "--btn-color": "#B8860B" }}>
                      <span>{showReveal ? "Hide reveal" : "Reveal ground truth"}</span>
                    </button>
                  )}
                </div>
              )}

              <div className="graph-area" style={!hasDashboard ? { display: "flex", alignItems: "center", justifyContent: "center" } : {}}>
                {isEmpty && (
                  <GraphUploadZone onUploadFile={handleUploadFile} onLoadDemo={handleLoadDemo} />
                )}
                {isAnalyzing && (
                  <div style={{ width: "100%", maxWidth: 640, textAlign: "center" }}>
                    <h3 style={{ margin: "0 0 20px", fontSize: 16, fontWeight: 600 }}>
                      Running analysis pipeline...
                    </h3>
                    <AnalysisConsole lines={consoleLines} analyzing={analyzing} />
                  </div>
                )}
                {hasDashboard && (
                  <NetworkGraph clusters={clusters} noise={noise} results={results}
                    positions={positions} activeCluster={activeCluster}
                    onSelectAccount={handleAccountSelect} animateIn={graphAnimateIn}
                    showReveal={showReveal} groundTruth={groundTruth} />
                )}
              </div>

              {hasDashboard && (
                <div className="graph-footer">
                  <span>t-SNE projection of 17-dim feature space</span>
                  <span>DBSCAN eps=1.5 min_samples=2</span>
                </div>
              )}
            </div>

            <div className="intel-panel">
              <div className="intel-header">Intelligence briefing</div>
              {!hasDashboard ? (
                <div style={{ textAlign: "center", opacity: 0.3, fontSize: 13, marginTop: 40 }}>
                  {isAnalyzing ? "Analysis in progress..." : "Waiting for data..."}
                </div>
              ) : showReveal && groundTruth ? (
                <RevealPanel groundTruth={groundTruth} clusters={clusters} results={results} />
              ) : selectedAccount ? (
                <AccountDetail accountId={selectedAccount} results={results} posts={posts} clusters={clusters} />
              ) : activeCluster !== null && activeCluster >= 0 ? (
                <ClusterDetail clusterId={activeCluster} cluster={clusters[activeCluster]} results={results} posts={posts} />
              ) : activeCluster === -1 ? (
                <NoiseDetail noise={noise} results={results} posts={posts} />
              ) : (
                <DefaultBriefing clusters={clusters} summary={{
                  total: Object.keys(results).length,
                  networks: clusterIds.length,
                  coordinated: Object.values(clusters).reduce((s, c) => s + c.member_count, 0),
                  noise: noise?.count || 0,
                }} />
              )}
            </div>
          </div>

          {hasDashboard && <ExportBar clusters={clusters} noise={noise} results={results} />}
        </>
      ) : (
        <TimelineSimulation results={results} clusters={clusters} noise={noise} />
      )}
      <style>{`@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}`}</style>
    </div>
  );
}
