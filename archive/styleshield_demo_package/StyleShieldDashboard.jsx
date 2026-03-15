import { useState, useEffect, useRef } from "react";

// ============================================================
// DATA — Replace with live API output from Caleb's scorer
// ============================================================
const DATASETS = {
  combined: {
    name: "Combined analysis",
    description: "GPT-4 bots + Haiku bots + real human airline tweets",
    summary: {
      accounts_analyzed: 34,
      bot_networks: 2,
      coordinated_accounts: 24,
      human_accounts: 10,
    },
    clusters: [
      {
        id: 0,
        label: "GPT-4 bot farm",
        color: "#E24B4A",
        isBot: true,
        member_count: 10,
        coordination_signal: 0.952,
        inferred_model: "GPT-4 / GPT-4o",
        inferred_timezone: "UTC-5 (EST)",
        shift_pattern: "09:00-11:00 (business hours)",
        metrics: { structural: 0.849, hedging: 0.512, vocabulary: 0.410, bot_score: 0.667, confidence: 0.811 },
        binding_features: ["intra_vocab_variance", "structural_regularity", "hedging_signature"],
        sample_text: "Certainly! This product offers exceptional value. Furthermore, it demonstrates notable quality in every aspect.",
        members: Array.from({ length: 10 }, (_, i) => ({
          id: `gpt4_account_${String(i).padStart(2, "0")}`,
          bot_score: 0.667,
          confidence: 0.811,
        })),
      },
      {
        id: 1,
        label: "Haiku bot farm",
        color: "#378ADD",
        isBot: true,
        member_count: 14,
        coordination_signal: 0.844,
        inferred_model: "Claude Haiku",
        inferred_timezone: "UTC-5 to UTC-8",
        shift_pattern: "08:00-22:00 (extended operation)",
        metrics: { structural: 0.812, hedging: 0.344, vocabulary: 0.336, bot_score: 0.617, confidence: 0.756 },
        binding_features: ["post_length_variance", "paragraph_rhythm_score", "transition_rate"],
        sample_text: "Just got the new Sony WH-1000XM5 headphones and wow, the noise cancellation is absolutely incredible. Best purchase I've made all year!",
        campaigns: [
          { name: "Product shills", count: 10, color: "#378ADD" },
          { name: "Geopolitical disinfo", count: 10, color: "#534AB7" },
          { name: "AI safety dismissal", count: 10, color: "#1D9E75" },
        ],
        members: [
          "TechGuru_Sam42", "gadget_dave91", "fit_life_ray", "wellness_jess88",
          "skin_glow_queen", "smart_shopper_nt", "RealTruth_News77",
          "geofacts_daily99", "patriot_eagle_55", "wkp_truth_seeker",
          "ai_progress_now", "free_think_ai", "build_dont_brake", "innovate_first_tk",
        ].map((name) => ({
          id: `haiku_${name}`,
          bot_score: 0.58 + Math.random() * 0.1,
          confidence: 0.70 + Math.random() * 0.1,
        })),
      },
    ],
    noise_accounts: [
      { id: "human_kbleggett", bot_score: 0.21, confidence: 0.18, sample: "@united so 8 hotels for 32 people but feel like we are being held hostage" },
      { id: "human_DatingRev", bot_score: 0.31, confidence: 0.22, sample: "@SouthwestAir your customer service is amazing as always, thanks for the help!" },
      { id: "human_NoviceFlyer", bot_score: 0.18, confidence: 0.15, sample: "First time flying @JetBlue and honestly impressed. Legroom is real" },
      { id: "human_Flora_Lola_NYC", bot_score: 0.27, confidence: 0.20, sample: "@united I've been on hold for 2 hrs. this is ridiculous. #neveragain" },
      { id: "human_pokecrastinator", bot_score: 0.35, confidence: 0.25, sample: "lol @AmericanAir lost my bag AGAIN. at this point it's a tradition" },
      { id: "human_LisaKothari", bot_score: 0.22, confidence: 0.17, sample: "@Delta thank you for the upgrade!! made my whole week" },
      { id: "human_MeereeneseKnot", bot_score: 0.29, confidence: 0.21, sample: "whoever designed the @SpiritAirlines app needs to fly spirit once" },
      { id: "human_DAngel082", bot_score: 0.19, confidence: 0.14, sample: "@united cancelling my flight 30 min before boarding is wild behavior" },
      { id: "human_pbpinftworth", bot_score: 0.33, confidence: 0.24, sample: "@SouthwestAir the wifi worked great on my flight today, shocking tbh" },
      { id: "human_Allisonjones704", bot_score: 0.25, confidence: 0.19, sample: "@JetBlue please help, stuck in BOS and nobody at the counter" },
    ],
  },
};

// ============================================================
// NETWORK GRAPH
// ============================================================
function NetworkGraph({ dataset, activeCluster }) {
  const canvasRef = useRef(null);
  const animRef = useRef(null);

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
    const clusterCount = dataset.clusters.length;

    dataset.clusters.forEach((cluster, ci) => {
      const cx = w * (0.18 + (ci / Math.max(clusterCount, 1)) * 0.4);
      const cy = h * 0.48;
      const radius = 22 + cluster.member_count * 3;

      cluster.members.forEach((member, mi) => {
        const angle = (mi / cluster.members.length) * Math.PI * 2 + ci * 0.5;
        const r = radius + (Math.random() - 0.5) * 10;
        nodes.push({
          x: cx + Math.cos(angle) * r,
          y: cy + Math.sin(angle) * r,
          clusterId: cluster.id,
          color: cluster.color,
          isBot: true,
          score: member.bot_score,
          id: member.id,
        });
      });
    });

    dataset.noise_accounts.forEach((human) => {
      nodes.push({
        x: w * 0.58 + Math.random() * w * 0.36,
        y: h * 0.1 + Math.random() * h * 0.8,
        clusterId: -1,
        color: "#1D9E75",
        isBot: false,
        score: human.bot_score,
        id: human.id,
      });
    });

    let frame = 0;
    function draw() {
      frame++;
      ctx.clearRect(0, 0, w, h);

      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const a = nodes[i], b = nodes[j];
          if (a.clusterId < 0 || b.clusterId < 0 || a.clusterId !== b.clusterId) continue;
          const dx = b.x - a.x, dy = b.y - a.y;
          if (Math.sqrt(dx * dx + dy * dy) > 130) continue;
          const dimmed = activeCluster !== null && a.clusterId !== activeCluster;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.strokeStyle = a.color;
          ctx.globalAlpha = dimmed ? 0.015 : 0.1;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
      ctx.globalAlpha = 1;

      nodes.forEach((n) => {
        const isActive = activeCluster === null || n.clusterId === activeCluster || (!n.isBot && activeCluster === -1);
        const pulse = Math.sin(frame * 0.025 + n.x * 0.01) * 0.3 + 0.7;
        const size = isActive ? (n.isBot ? 5 : 3.5) : 2;
        ctx.globalAlpha = isActive ? 0.88 : 0.1;

        if (isActive && n.isBot) {
          ctx.beginPath();
          ctx.arc(n.x, n.y, size + 5 * pulse, 0, Math.PI * 2);
          ctx.fillStyle = n.color;
          ctx.globalAlpha = 0.06;
          ctx.fill();
          ctx.globalAlpha = isActive ? 0.88 : 0.1;
        }

        ctx.beginPath();
        ctx.arc(n.x, n.y, size, 0, Math.PI * 2);
        ctx.fillStyle = n.color;
        ctx.fill();
      });

      ctx.globalAlpha = 1;
      dataset.clusters.forEach((cluster, ci) => {
        const cx = w * (0.18 + (ci / Math.max(clusterCount, 1)) * 0.4);
        const radius = 22 + cluster.member_count * 3;
        const dimmed = activeCluster !== null && activeCluster !== cluster.id;
        ctx.globalAlpha = dimmed ? 0.15 : 0.85;
        ctx.font = "500 11px system-ui, -apple-system, sans-serif";
        ctx.fillStyle = cluster.color;
        ctx.textAlign = "center";
        ctx.fillText(cluster.label, cx, h * 0.48 - radius - 12);
        ctx.font = "400 10px system-ui, -apple-system, sans-serif";
        ctx.fillText(`${cluster.member_count} accounts`, cx, h * 0.48 - radius);
      });

      const humanDimmed = activeCluster !== null && activeCluster !== -1;
      ctx.globalAlpha = humanDimmed ? 0.15 : 0.7;
      ctx.font = "500 11px system-ui, -apple-system, sans-serif";
      ctx.fillStyle = "#1D9E75";
      ctx.textAlign = "center";
      ctx.fillText("Human accounts (noise)", w * 0.76, h * 0.06);
      ctx.font = "400 10px system-ui, -apple-system, sans-serif";
      ctx.fillText(`${dataset.noise_accounts.length} accounts \u00b7 no coordination`, w * 0.76, h * 0.06 + 14);
      ctx.globalAlpha = 1;

      animRef.current = requestAnimationFrame(draw);
    }
    draw();
    return () => cancelAnimationFrame(animRef.current);
  }, [dataset, activeCluster]);

  return <canvas ref={canvasRef} style={{ width: "100%", height: "100%", display: "block" }} />;
}

// ============================================================
// SMALL COMPONENTS
// ============================================================
function MetricBar({ label, value, color }) {
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 2 }}>
        <span style={{ fontSize: 11, opacity: 0.6, textTransform: "uppercase", letterSpacing: 0.5 }}>{label}</span>
        <span style={{ fontSize: 11, fontFamily: "monospace" }}>{(value * 100).toFixed(1)}%</span>
      </div>
      <div style={{ height: 4, background: "rgba(128,128,128,0.15)", borderRadius: 2, overflow: "hidden" }}>
        <div style={{ height: "100%", width: `${value * 100}%`, background: color, borderRadius: 2, transition: "width 0.6s ease-out" }} />
      </div>
    </div>
  );
}

function Stamp({ text, color }) {
  return (
    <span style={{ display: "inline-block", padding: "2px 8px", border: `1px solid ${color}`, color, fontSize: 10, fontFamily: "monospace", textTransform: "uppercase", letterSpacing: 1, borderRadius: 4, transform: "rotate(-1deg)", marginRight: 6 }}>
      {text}
    </span>
  );
}

function Card({ label, children }) {
  return (
    <div style={{ background: "rgba(128,128,128,0.06)", borderRadius: 8, padding: 14, marginBottom: 10, border: "1px solid rgba(128,128,128,0.1)" }}>
      {label && <div style={{ fontSize: 10, opacity: 0.4, textTransform: "uppercase", letterSpacing: 1.5, marginBottom: 8 }}>{label}</div>}
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
        <div key={h} style={{ fontSize: 8, opacity: 0.4, textAlign: "center", fontFamily: "monospace" }}>{h % 6 === 0 ? h : ""}</div>
      ))}
      {days.map((day, di) => (
        <div key={day} style={{ display: "contents" }}>
          <div style={{ fontSize: 9, opacity: 0.4, display: "flex", alignItems: "center" }}>{day}</div>
          {Array.from({ length: 24 }, (_, h) => {
            let intensity;
            if (isBot) {
              intensity = h >= 9 && h <= 17 && di < 5 ? 0.5 + Math.random() * 0.4 : Math.random() * 0.06;
            } else {
              intensity = ((h >= 18 && h <= 22) ? 0.3 : (h >= 7 && h <= 9) ? 0.25 : 0.03) + Math.random() * 0.12;
            }
            return <div key={h} style={{ height: 10, borderRadius: 1, background: `rgba(${isBot ? "226,75,74" : "29,158,117"}, ${intensity})` }} />;
          })}
        </div>
      ))}
    </div>
  );
}

// ============================================================
// INTEL PANELS
// ============================================================
function ClusterDetail({ cluster }) {
  const c = cluster, m = c.metrics;
  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
        <div style={{ width: 12, height: 12, borderRadius: "50%", background: c.color }} />
        <span style={{ fontSize: 16, fontWeight: 600 }}>{c.label}</span>
      </div>
      <div style={{ marginBottom: 14 }}>
        <Stamp text="coordinated" color="#E24B4A" />
        <Stamp text="ai generated" color="#D85A30" />
      </div>
      <Card label="Infrastructure profile">
        {[["Inferred model", c.inferred_model], ["Time zone", c.inferred_timezone], ["Shift pattern", c.shift_pattern], ["Accounts", `${c.member_count} detected`], ["Coordination", `${(c.coordination_signal * 100).toFixed(1)}%`]].map(([l, v]) => (
          <div key={l} style={{ display: "flex", justifyContent: "space-between", padding: "4px 0", borderBottom: "1px solid rgba(128,128,128,0.08)", fontSize: 12 }}>
            <span style={{ opacity: 0.5 }}>{l}</span><span style={{ fontFamily: "monospace" }}>{v}</span>
          </div>
        ))}
      </Card>
      <Card label="Stylometric signatures">
        <MetricBar label="Structural regularity" value={m.structural} color={c.color} />
        <MetricBar label="Hedging signature" value={m.hedging} color={c.color} />
        <MetricBar label="Vocabulary uniformity" value={m.vocabulary} color={c.color} />
        <MetricBar label="Bot score" value={m.bot_score} color={m.bot_score > 0.5 ? "#E24B4A" : "#1D9E75"} />
        <MetricBar label="Confidence" value={m.confidence} color={m.confidence > 0.7 ? "#D85A30" : "#378ADD"} />
      </Card>
      <Card label="Posting pattern">
        <TemporalHeatmap isBot={true} />
      </Card>
      {c.campaigns && (
        <Card label="Campaign breakdown">
          {c.campaigns.map((camp) => (
            <div key={camp.name} style={{ display: "flex", alignItems: "center", gap: 8, padding: "3px 0", fontSize: 12 }}>
              <div style={{ width: 8, height: 8, borderRadius: 2, background: camp.color }} />
              <span>{camp.name}</span>
              <span style={{ marginLeft: "auto", fontFamily: "monospace", opacity: 0.5 }}>{camp.count}</span>
            </div>
          ))}
        </Card>
      )}
      {c.binding_features.length > 0 && (
        <Card label="Binding features">
          <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
            {c.binding_features.map((f) => (
              <span key={f} style={{ padding: "2px 8px", fontSize: 10, fontFamily: "monospace", background: "rgba(128,128,128,0.1)", borderRadius: 4 }}>{f}</span>
            ))}
          </div>
        </Card>
      )}
      <Card label="Sample text">
        <div style={{ fontSize: 12, opacity: 0.6, fontStyle: "italic", lineHeight: 1.6, borderLeft: `2px solid ${c.color}`, paddingLeft: 12 }}>"{c.sample_text}"</div>
      </Card>
      <Card label={`Member accounts (${c.member_count})`}>
        <div style={{ maxHeight: 180, overflowY: "auto" }}>
          {c.members.map((m) => (
            <div key={m.id} style={{ display: "flex", justifyContent: "space-between", padding: "3px 0", fontSize: 11, fontFamily: "monospace", borderBottom: "1px solid rgba(128,128,128,0.06)" }}>
              <span style={{ opacity: 0.5 }}>{m.id}</span>
              <span style={{ color: m.confidence > 0.7 ? "#E24B4A" : "#D85A30" }}>{(m.confidence * 100).toFixed(0)}%</span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

function NoiseDetail({ accounts }) {
  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
        <div style={{ width: 12, height: 12, borderRadius: "50%", background: "#1D9E75" }} />
        <span style={{ fontSize: 16, fontWeight: 600 }}>Human accounts</span>
      </div>
      <div style={{ marginBottom: 14 }}>
        <Stamp text="organic" color="#1D9E75" />
        <Stamp text="no coordination" color="#1D9E75" />
      </div>
      <Card label="Assessment">
        <div style={{ fontSize: 12, opacity: 0.6, lineHeight: 1.7 }}>
          These {accounts.length} accounts exhibit natural human writing variance: inconsistent sentence structure, informal spelling, emotional variation across posts, and irregular posting patterns. No stylometric clustering detected. Each account has a distinct writing fingerprint.
        </div>
      </Card>
      <Card label="Posting pattern">
        <TemporalHeatmap isBot={false} />
      </Card>
      <Card label="Why they're not bots">
        {[["Structural regularity", "Low (0.42 avg) \u2014 humans vary wildly"], ["Intra-account variance", "High \u2014 writing shifts post to post"], ["Hedging markers", "Minimal \u2014 no LLM-characteristic phrases"], ["Cross-account correlation", "Low \u2014 each voice is distinct"]].map(([l, v]) => (
          <div key={l} style={{ padding: "4px 0", borderBottom: "1px solid rgba(128,128,128,0.06)", fontSize: 12 }}>
            <div style={{ fontWeight: 500, marginBottom: 2 }}>{l}</div>
            <div style={{ opacity: 0.5 }}>{v}</div>
          </div>
        ))}
      </Card>
      <Card label={`Accounts (${accounts.length})`}>
        {accounts.map((a) => (
          <div key={a.id} style={{ padding: "6px 0", borderBottom: "1px solid rgba(128,128,128,0.06)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, fontFamily: "monospace", marginBottom: 4 }}>
              <span style={{ opacity: 0.5 }}>{a.id}</span>
              <span style={{ color: "#1D9E75" }}>bot: {(a.bot_score * 100).toFixed(0)}%</span>
            </div>
            <div style={{ fontSize: 11, opacity: 0.4, fontStyle: "italic" }}>"{a.sample}"</div>
          </div>
        ))}
      </Card>
    </div>
  );
}

function DefaultBriefing({ summary }) {
  return (
    <div>
      <Card label="Summary">
        <div style={{ fontSize: 13, lineHeight: 1.8, opacity: 0.7 }}>
          Analysis of <strong style={{ opacity: 1 }}>{summary.accounts_analyzed} accounts</strong> reveals{" "}
          <strong style={{ color: "#E24B4A", opacity: 1 }}>{summary.bot_networks} coordinated bot networks</strong>{" "}
          comprising <strong style={{ opacity: 1 }}>{summary.coordinated_accounts} accounts</strong>.{" "}
          <strong style={{ color: "#1D9E75", opacity: 1 }}>{summary.human_accounts} accounts</strong>{" "}
          classified as organic (noise).
        </div>
      </Card>
      <Card label="Key finding">
        <div style={{ fontSize: 12, lineHeight: 1.7, opacity: 0.6 }}>
          Two distinct AI model signatures detected. GPT-4 accounts show characteristic hedging ("certainly," "furthermore") with high structural regularity (0.849). Haiku accounts show lower hedging but uniform paragraph rhythm across 3 campaigns: product shills, geopolitical disinfo, AI safety dismissal. Human accounts scatter with no correlation.
        </div>
      </Card>
      <Card label="Xenarch methodology">
        <div style={{ fontSize: 12, lineHeight: 1.7, opacity: 0.6 }}>
          Adapted from planetary technosignature detection. Xenarch finds artificial structures in natural geology (99.58% confidence on Apollo 11 lander). StyleShield finds artificial coordination in natural discourse. Five normalized metrics, DBSCAN clustering on cosine distance in 18-dim feature space. Same math, inverse mission.
        </div>
      </Card>
      <div style={{ fontSize: 12, opacity: 0.35, marginTop: 16, textAlign: "center" }}>Select a cluster to inspect</div>
    </div>
  );
}

// ============================================================
// MAIN DASHBOARD
// ============================================================
export default function StyleShieldDashboard() {
  const [activeCluster, setActiveCluster] = useState(null);
  const dataset = DATASETS.combined;
  const handleSelect = (id) => setActiveCluster(activeCluster === id ? null : id);
  const activeData = activeCluster !== null && activeCluster >= 0 ? dataset.clusters.find((c) => c.id === activeCluster) : null;

  return (
    <div style={{ minHeight: "100vh", padding: 16, fontFamily: "system-ui, -apple-system, sans-serif" }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
        <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#E24B4A", boxShadow: "0 0 6px rgba(226,75,74,0.5)", animation: "pulse 2s infinite" }} />
        <span style={{ fontSize: 16, fontWeight: 600, letterSpacing: 2, textTransform: "uppercase" }}>StyleShield</span>
        <span style={{ fontSize: 12, opacity: 0.4, letterSpacing: 1 }}>bot network detection</span>
      </div>

      {/* Stats */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8, marginBottom: 16 }}>
        {[
          { label: "Accounts scanned", value: dataset.summary.accounts_analyzed, color: "inherit" },
          { label: "Networks detected", value: dataset.summary.bot_networks, color: "#E24B4A" },
          { label: "Coordinated", value: dataset.summary.coordinated_accounts, color: "#D85A30" },
          { label: "Humans (noise)", value: dataset.summary.human_accounts, color: "#1D9E75" },
        ].map(({ label, value, color }) => (
          <div key={label} style={{ background: "rgba(128,128,128,0.06)", borderRadius: 8, padding: "10px 14px" }}>
            <div style={{ fontSize: 10, opacity: 0.4, textTransform: "uppercase", letterSpacing: 1, marginBottom: 4 }}>{label}</div>
            <div style={{ fontSize: 26, fontWeight: 400, fontFamily: "monospace", color }}>{value}</div>
          </div>
        ))}
      </div>

      {/* Main grid */}
      <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) 320px", gap: 12 }}>
        {/* Graph */}
        <div style={{ background: "rgba(128,128,128,0.04)", borderRadius: 12, padding: 12, position: "relative", minHeight: 420 }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 4, marginBottom: 8 }}>
            {dataset.clusters.map((cluster) => (
              <button key={cluster.id} onClick={() => handleSelect(cluster.id)} style={{ display: "flex", alignItems: "center", gap: 8, padding: "5px 10px", background: activeCluster === cluster.id ? "rgba(128,128,128,0.1)" : "transparent", border: activeCluster === cluster.id ? `1px solid ${cluster.color}40` : "1px solid transparent", borderRadius: 6, cursor: "pointer", fontSize: 12, color: "inherit", textAlign: "left" }}>
                <div style={{ width: 10, height: 10, borderRadius: "50%", background: cluster.color, flexShrink: 0 }} />
                <span>{cluster.label}</span>
                <span style={{ marginLeft: "auto", fontSize: 10, fontFamily: "monospace", opacity: 0.4 }}>{cluster.member_count}</span>
              </button>
            ))}
            <button onClick={() => handleSelect(-1)} style={{ display: "flex", alignItems: "center", gap: 8, padding: "5px 10px", background: activeCluster === -1 ? "rgba(128,128,128,0.1)" : "transparent", border: activeCluster === -1 ? "1px solid rgba(29,158,117,0.4)" : "1px solid transparent", borderRadius: 6, cursor: "pointer", fontSize: 12, color: "inherit", textAlign: "left" }}>
              <div style={{ width: 10, height: 10, borderRadius: "50%", background: "#1D9E75", flexShrink: 0 }} />
              <span>Human accounts (noise)</span>
              <span style={{ marginLeft: "auto", fontSize: 10, fontFamily: "monospace", opacity: 0.4 }}>{dataset.noise_accounts.length}</span>
            </button>
          </div>
          <div style={{ height: 340 }}>
            <NetworkGraph dataset={dataset} activeCluster={activeCluster} />
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, fontFamily: "monospace", opacity: 0.3, marginTop: 8 }}>
            <span>DBSCAN e=0.08 min_samples=2</span>
            <span>cosine similarity 18-dim</span>
          </div>
        </div>

        {/* Intel */}
        <div style={{ background: "rgba(128,128,128,0.04)", borderRadius: 12, padding: 16, maxHeight: 540, overflowY: "auto" }}>
          <div style={{ fontSize: 10, opacity: 0.4, textTransform: "uppercase", letterSpacing: 1.5, marginBottom: 12 }}>Intelligence briefing</div>
          {activeData ? <ClusterDetail cluster={activeData} /> : activeCluster === -1 ? <NoiseDetail accounts={dataset.noise_accounts} /> : <DefaultBriefing summary={dataset.summary} />}
        </div>
      </div>

      <style>{`@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}} ::-webkit-scrollbar{width:5px} ::-webkit-scrollbar-thumb{background:rgba(128,128,128,0.2);border-radius:3px}`}</style>
    </div>
  );
}
