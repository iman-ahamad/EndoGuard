import { useState, useEffect, useRef } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  LineChart, Line, ResponsiveContainer,
} from "recharts";

// ── Config ───────────────────────────────────────────────────
// Keep deployment-specific values in frontend/.env (never commit that file).
const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");
const HEADERS = import.meta.env.VITE_NGROK_SKIP_BROWSER_WARNING === "false"
  ? {}
  : { "ngrok-skip-browser-warning": "true" };

const ENTITY_COLORS = {
  polyp: { color: "#ef4444", type: "abnormality" },
  "ulcerative colitis": { color: "#f97316", type: "abnormality" },
  oesophagitis: { color: "#eab308", type: "abnormality" },
  "z-line": { color: "#3b82f6", type: "landmark" },
  pylorus: { color: "#3b82f6", type: "landmark" },
  cecum: { color: "#3b82f6", type: "landmark" },
  tube: { color: "#22c55e", type: "instrument" },
  "polyp snare": { color: "#22c55e", type: "instrument" },
  "biopsy forceps": { color: "#22c55e", type: "instrument" },
};

const SUGGESTED_QS = [
  "What do you see in this endoscopy image?",
  "Are there any polyps visible?",
  "Describe the mucosal surface.",
  "Is this image normal or abnormal?",
  "What anatomical landmark is shown?",
];

// ── Theme ────────────────────────────────────────────────────
const themes = {
  dark: {
    bg: "#080d18", bgCard: "rgba(15,23,42,0.6)", bgCardSolid: "#0f172a",
    text: "#e2e8f0", textMuted: "#94a3b8", textHeading: "#f8fafc",
    accent: "#00c9a7", accentAlt: "#38bdf8", border: "rgba(148,163,184,0.15)",
    inputBg: "rgba(15,23,42,0.8)", successBg: "rgba(0,201,167,0.12)",
    dangerBg: "rgba(239,68,68,0.12)", warnBg: "rgba(249,115,22,0.12)",
  },
  light: {
    bg: "hsla(44, 19%, 76%, 0.78)", bgCard: "rgba(255,255,255,0.7)", bgCardSolid: "#ffffff",
    text: "#1e293b", textMuted: "#64748b", textHeading: "#0f172a",
    accent: "#00897b", accentAlt: "#0277bd", border: "rgba(0,0,0,0.1)",
    inputBg: "rgba(255,255,255,0.9)", successBg: "rgba(0,137,123,0.1)",
    dangerBg: "rgba(239,68,68,0.1)", warnBg: "rgba(249,115,22,0.1)",
  },
};

// ── Fonts (injected once) ────────────────────────────────────
const fontLink = document.createElement("link");
fontLink.rel = "stylesheet";
fontLink.href =
  "https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,600;0,700&family=DM+Serif+Display&family=Fira+Code:wght@400;500&display=swap";
if (!document.querySelector('link[href*="DM+Sans"]'))
  document.head.appendChild(fontLink);

// ── Helpers ──────────────────────────────────────────────────
const api = async (path, opts = {}) => {
  const res = await fetch(`${API_BASE}${path}`, { ...opts, headers: { ...HEADERS, ...opts.headers } });
  if (!res.ok) throw new Error(`API ${res.status}`);
  return res.json();
};

// Spelling aliases for entity matching
// Spelling aliases for entity matching — all variants that should highlight as the canonical entity
const ENTITY_ALIAS_MAP = {
  oesophagitis: [
    "oesophagitis", "esophagitis",
    "oesophageal", "esophageal",
    "erosive esophagitis", "erosive oesophagitis",
    "reflux esophagitis", "reflux oesophagitis",
  ],
  "ulcerative colitis": ["ulcerative colitis", "ulcerative-colitis"],
  polyp: ["polyp"],
};

function highlightEntities(text, entities, flagged = []) {
  if (!text || !entities?.length) return text;
  const sorted = [...entities].sort((a, b) => b.length - a.length);
  let result = text;
  for (const ent of sorted) {
    const c = ENTITY_COLORS[ent]?.color || "#9ca3af";
    const isFlagged = flagged.includes(ent);
    // Build pattern that includes spelling variants
    const variants = ENTITY_ALIAS_MAP[ent] || [ent];
    const sortedVariants = [...variants].sort((a, b) => b.length - a.length);
    const patStr = sortedVariants.map(v => v.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "s?").join("|");
    const re = new RegExp(`\\b(${patStr})\\b`, "gi");
    result = result.replace(re, (m) => {
      const style = isFlagged
        ? `color:${c};text-decoration:line-through;text-decoration-color:${c};opacity:0.7`
        : `color:${c};font-weight:600`;
      return `<span style="${style}">${m}</span>`;
    });
  }
  return result;
}

// ── Icons (inline SVG) ───────────────────────────────────────
const Icon = ({ name, size = 18, color = "currentColor" }) => {
  const s = { width: size, height: size, display: "inline-block", flexShrink: 0 };
  const svgProps = { width: size, height: size, viewBox: "0 0 24 24", fill: "none", stroke: color, strokeWidth: 2, strokeLinecap: "round", strokeLinejoin: "round" };
  const icons = {
    shield: <svg {...svgProps}><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" /></svg>,
    check: <svg {...svgProps}><polyline points="20 6 9 17 4 12" /></svg>,
    alert: <svg {...svgProps}><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" /><line x1="12" y1="9" x2="12" y2="13" /><line x1="12" y1="17" x2="12.01" y2="17" /></svg>,
    help: <svg {...svgProps}><circle cx="12" cy="12" r="10" /><path d="M9.09 9a3 3 0 015.83 1c0 2-3 3-3 3" /><line x1="12" y1="17" x2="12.01" y2="17" /></svg>,
    sun: <svg {...svgProps}><circle cx="12" cy="12" r="5" /><line x1="12" y1="1" x2="12" y2="3" /><line x1="12" y1="21" x2="12" y2="23" /><line x1="4.22" y1="4.22" x2="5.64" y2="5.64" /><line x1="18.36" y1="18.36" x2="19.78" y2="19.78" /><line x1="1" y1="12" x2="3" y2="12" /><line x1="21" y1="12" x2="23" y2="12" /><line x1="4.22" y1="19.78" x2="5.64" y2="18.36" /><line x1="18.36" y1="5.64" x2="19.78" y2="4.22" /></svg>,
    moon: <svg {...svgProps}><path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z" /></svg>,
    upload: <svg {...svgProps}><polyline points="16 16 12 12 8 16" /><line x1="12" y1="12" x2="12" y2="21" /><path d="M20.39 18.39A5 5 0 0018 9h-1.26A8 8 0 103 16.3" /></svg>,
    send: <svg {...svgProps}><line x1="22" y1="2" x2="11" y2="13" /><polygon points="22 2 15 22 11 13 2 9 22 2" /></svg>,
    loader: <svg {...svgProps} style={{ ...s, animation: "spin 1s linear infinite" }}><line x1="12" y1="2" x2="12" y2="6" /><line x1="12" y1="18" x2="12" y2="22" /><line x1="4.93" y1="4.93" x2="7.76" y2="7.76" /><line x1="16.24" y1="16.24" x2="19.07" y2="19.07" /><line x1="2" y1="12" x2="6" y2="12" /><line x1="18" y1="12" x2="22" y2="12" /><line x1="4.93" y1="19.07" x2="7.76" y2="16.24" /><line x1="16.24" y1="7.76" x2="19.07" y2="4.93" /></svg>,
    file: <svg {...svgProps}><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" /><polyline points="14 2 14 8 20 8" /></svg>,
    arrow: <svg {...svgProps}><line x1="5" y1="12" x2="19" y2="12" /><polyline points="12 5 19 12 12 19" /></svg>,
    home: <svg {...svgProps}><path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z" /><polyline points="9 22 9 12 15 12 15 22" /></svg>,
    search: <svg {...svgProps}><circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" /></svg>,
    book: <svg {...svgProps}><path d="M2 3h6a4 4 0 014 4v14a3 3 0 00-3-3H2z" /><path d="M22 3h-6a4 4 0 00-4 4v14a3 3 0 013-3h7z" /></svg>,
    download: <svg {...svgProps}><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" /><polyline points="7 10 12 15 17 10" /><line x1="12" y1="15" x2="12" y2="3" /></svg>,
  };
  return <span style={s}>{icons[name] || null}</span>;
};

// ── Main App ─────────────────────────────────────────────────
export default function EndoGuard() {
  const [page, setPage] = useState("home");
  const [theme, setTheme] = useState("dark");
  const [mode, setMode] = useState("dpo_only");
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [selectedGalleryId, setSelectedGalleryId] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [gallery, setGallery] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [connected, setConnected] = useState(false);
  const [qa, setQa] = useState([]);
  const [qaInput, setQaInput] = useState("");
  const [qaLoading, setQaLoading] = useState(false);
  const [showReport, setShowReport] = useState(false);
  const [expandedPhase, setExpandedPhase] = useState(null);
  const timerRef = useRef(null);
  const fileRef = useRef(null);

  const t = themes[theme];

  // ── Health check ──
  useEffect(() => {
    const check = async () => {
      try {
        await api("/health");
        setConnected(true);
      } catch { setConnected(false); }
    };
    check();
    const iv = setInterval(check, 30000);
    return () => clearInterval(iv);
  }, []);

  // ── Load gallery + metrics ──
  useEffect(() => {
    if (!connected) return;
    api("/gallery").then(async (d) => {
      const images = d.images || [];
      // Fetch base64 thumbnails for each gallery image
      const withThumbs = await Promise.all(
        images.map(async (img) => {
          try {
            const b64 = await api(`/gallery/${img.id}/base64`);
            return { ...img, thumb: b64.data_uri };
          } catch {
            return { ...img, thumb: null };
          }
        })
      );
      setGallery(withThumbs);
    }).catch(() => { });
    api("/metrics").then(d => setMetrics(d)).catch(() => { });
  }, [connected]);

  // ── Timer during loading ──
  useEffect(() => {
    if (loading) {
      setElapsed(0);
      timerRef.current = setInterval(() => setElapsed(p => p + 0.1), 100);
    } else {
      clearInterval(timerRef.current);
    }
    return () => clearInterval(timerRef.current);
  }, [loading]);

  // ── Upload handler ──
  const handleFile = (file) => {
    if (!file) return;
    setImage(file);
    setSelectedGalleryId(null);
    setResult(null);
    setQa([]);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target.result);
    reader.readAsDataURL(file);
  };

  // ── Gallery select ──
  const selectGallery = (item) => {
    setSelectedGalleryId(item.id);
    setImage(null);
    setPreview(item.thumb || `${API_BASE}/gallery/${item.id}`);
    setResult(null);
    setQa([]);
  };

  // ── Reset for new analysis ──
  const resetAnalysis = () => {
    setImage(null);
    setPreview(null);
    setSelectedGalleryId(null);
    setResult(null);
    setQa([]);
    setQaInput("");
    setLoading(false);
    if (fileRef.current) fileRef.current.value = "";
  };

  // ── Analyze ──
  const runAnalysis = async (question = "What do you see in this endoscopy image?") => {
    if (!image && !selectedGalleryId) return;
    setLoading(true);
    setResult(null);
    try {
      const fd = new FormData();
      if (image) fd.append("image", image);
      else fd.append("image_id", selectedGalleryId);
      fd.append("mode", mode);
      fd.append("question", question);
      const data = await fetch(`${API_BASE}/analyze`, { method: "POST", headers: HEADERS, body: fd }).then(r => {
        if (!r.ok) throw new Error(`API ${r.status}`);
        return r.json();
      });
      setResult(data);
    } catch (e) {
      setResult({ error: e.message || "Analysis failed" });
    } finally {
      setLoading(false);
    }
  };

  // ── Follow-up QA ──
  const askFollowUp = async (q) => {
    if (!q.trim() || (!image && !selectedGalleryId)) return;
    setQaLoading(true);
    try {
      const fd = new FormData();
      if (image) fd.append("image", image);
      else fd.append("image_id", selectedGalleryId);
      fd.append("mode", mode);
      fd.append("question", q);
      const data = await fetch(`${API_BASE}/analyze`, { method: "POST", headers: HEADERS, body: fd }).then(r => r.json());
      setQa(prev => [...prev, { q, a: data.response || data.final_response || data.initial_response || "No response", data }]);
    } catch { setQa(prev => [...prev, { q, a: "Error getting response" }]); }
    finally { setQaLoading(false); setQaInput(""); }
  };

  // ── Styles ──
  const cardStyle = {
    background: t.bgCard, backdropFilter: "blur(16px)", border: `1px solid ${t.border}`,
    borderRadius: 14, padding: "1.25rem", transition: "all 0.2s",
  };
  const btnPrimary = {
    background: `linear-gradient(135deg, ${t.accent}, ${t.accentAlt})`, color: "#fff",
    border: "none", borderRadius: 10, padding: "0.65rem 1.5rem", cursor: "pointer",
    fontFamily: "'DM Sans',sans-serif", fontWeight: 600, fontSize: 14, transition: "all 0.2s",
  };
  const btnOutline = {
    background: "transparent", color: t.accent, border: `1px solid ${t.accent}`,
    borderRadius: 10, padding: "0.5rem 1.2rem", cursor: "pointer",
    fontFamily: "'DM Sans',sans-serif", fontWeight: 500, fontSize: 13, transition: "all 0.2s",
  };

  // ════════════════════════════════════════════════════════════
  // RENDER
  // ════════════════════════════════════════════════════════════
  return (
    <div style={{
      fontFamily: "'DM Sans',sans-serif", color: t.text, background: t.bg,
      minHeight: "100vh", transition: "background 0.3s, color 0.3s",
      position: "relative",
    }}>
      <ParticleBackground theme={theme} />
      <style>{`

        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }
        .fade-in { animation: fadeIn 0.5s ease-out both; }
        .hover-lift:hover { transform: translateY(-3px); box-shadow: 0 8px 30px rgba(0,0,0,0.2); }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-thumb { background: ${t.border}; border-radius: 3px; }
        @media print { .no-print { display: none !important; } }
      `}</style>

      {/* ── Header ── */}
      <header className="no-print" style={{
        position: "sticky", top: 0, zIndex: 100, background: t.bgCard,
        backdropFilter: "blur(20px)", borderBottom: `1px solid ${t.border}`,
        padding: "0 1.5rem", display: "flex", alignItems: "center", justifyContent: "space-between",
        height: 56,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer" }} onClick={() => setPage("home")}>
          <Icon name="shield" size={22} color={t.accent} />
          <span style={{ fontFamily: "'DM Serif Display',serif", fontSize: 20, color: t.textHeading }}>EndoGuard</span>
        </div>
        <nav style={{ display: "flex", alignItems: "center", gap: 4 }}>
          {[["home", "Home", <Icon name="home" size={15} />], ["analyze", "Analyze", <Icon name="search" size={15} />], ["methodology", "Method", <Icon name="book" size={15} />]].map(([id, label, icon]) => (
            <button key={id} onClick={() => setPage(id)} style={{
              background: page === id ? `${t.accent}18` : "transparent", color: page === id ? t.accent : t.textMuted,
              border: "none", borderRadius: 8, padding: "6px 14px", cursor: "pointer",
              fontFamily: "'DM Sans'", fontWeight: 500, fontSize: 13, display: "flex", alignItems: "center", gap: 5,
              transition: "all 0.2s",
            }}>{icon}{label}</button>
          ))}
        </nav>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 12, color: t.textMuted }}>
            <span style={{
              width: 7, height: 7, borderRadius: "50%",
              background: connected ? "#22c55e" : "#ef4444",
              boxShadow: connected ? "0 0 6px #22c55e" : "0 0 6px #ef4444",
            }} />
            {connected ? "Connected" : "Offline"}
          </div>
          <button onClick={() => setTheme(theme === "dark" ? "light" : "dark")} style={{
            background: "transparent", border: `1px solid ${t.border}`, borderRadius: 8,
            padding: "5px 8px", cursor: "pointer", color: t.textMuted, display: "flex",
          }}>
            <Icon name={theme === "dark" ? "sun" : "moon"} size={16} />
          </button>
        </div>
      </header>

      {/* ── Connection banner ── */}
      {!connected && (
        <div className="no-print" style={{
          background: "#ef444420", borderBottom: "1px solid #ef444440", padding: "8px 1.5rem",
          display: "flex", alignItems: "center", justifyContent: "center", gap: 12, fontSize: 13,
        }}>
          <span>⚠ Cannot reach EndoGuard API. Check VITE_API_BASE_URL in your .env file.</span>
          <button onClick={async () => { try { await api("/health"); setConnected(true); } catch { } }}
            style={{ ...btnOutline, padding: "3px 12px", fontSize: 12, color: "#ef4444", borderColor: "#ef4444" }}>
            Retry
          </button>
        </div>
      )}

      <main style={{ maxWidth: 1200, margin: "0 auto", padding: "0 1.5rem 3rem", position: "relative", zIndex: 1 }}>
        {page === "home" && <HomePage t={t} metrics={metrics} setPage={setPage} cardStyle={cardStyle} btnPrimary={btnPrimary} btnOutline={btnOutline} gallery={gallery} selectGallery={selectGallery} />}
        {page === "analyze" && (
          <AnalyzePage t={t} mode={mode} setMode={setMode} image={image} preview={preview}
            selectedGalleryId={selectedGalleryId} gallery={gallery} result={result}
            loading={loading} elapsed={elapsed} qa={qa} qaInput={qaInput} qaLoading={qaLoading}
            showReport={showReport} fileRef={fileRef} cardStyle={cardStyle} btnPrimary={btnPrimary}
            btnOutline={btnOutline} handleFile={handleFile} selectGallery={selectGallery}
            runAnalysis={runAnalysis} askFollowUp={askFollowUp} setQaInput={setQaInput}
            setShowReport={setShowReport} resetAnalysis={resetAnalysis} />
        )}
        {page === "methodology" && <MethodologyPage t={t} cardStyle={cardStyle} expandedPhase={expandedPhase} setExpandedPhase={setExpandedPhase} metrics={metrics} />}
      </main>
    </div>
  );
}

// ── Particle Network Background ──────────────────────────────
function ParticleBackground({ theme }) {
  const canvasRef = useRef(null);
  const isDark = theme === "dark";

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    let animId;
    let particles = [];
    let mouse = { x: -9999, y: -9999 };

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener("resize", resize);

    const handleMouse = (e) => { mouse.x = e.clientX; mouse.y = e.clientY; };
    window.addEventListener("mousemove", handleMouse);

    const COUNT = Math.min(Math.floor((window.innerWidth * window.innerHeight) / 18000), 70);
    const CONNECT_DIST = 140;
    const MOUSE_DIST = 180;

    const colors = isDark
      ? ["rgba(0,201,167,", "rgba(56,189,248,", "rgba(168,85,247,"]
      : ["rgba(0,137,123,", "rgba(2,119,189,", "rgba(139,92,246,"];

    class Particle {
      constructor() {
        this.x = Math.random() * canvas.width;
        this.y = Math.random() * canvas.height;
        this.vx = (Math.random() - 0.5) * 0.35;
        this.vy = (Math.random() - 0.5) * 0.35;
        this.r = Math.random() * 2 + 1;
        this.colorIdx = Math.floor(Math.random() * 3);
        this.baseAlpha = Math.random() * 0.4 + 0.15;
        this.pulseOffset = Math.random() * Math.PI * 2;
        this.pulseSpeed = 0.005 + Math.random() * 0.01;
      }
      update(t) {
        this.x += this.vx;
        this.y += this.vy;
        if (this.x < 0 || this.x > canvas.width) this.vx *= -1;
        if (this.y < 0 || this.y > canvas.height) this.vy *= -1;
        this.alpha = this.baseAlpha + Math.sin(t * this.pulseSpeed + this.pulseOffset) * 0.1;
      }
      draw() {
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.r, 0, Math.PI * 2);
        ctx.fillStyle = colors[this.colorIdx] + this.alpha + ")";
        ctx.fill();
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.r * 3, 0, Math.PI * 2);
        ctx.fillStyle = colors[this.colorIdx] + (this.alpha * 0.15) + ")";
        ctx.fill();
      }
    }

    for (let i = 0; i < COUNT; i++) particles.push(new Particle());

    let t = 0;
    const animate = () => {
      t++;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      for (const p of particles) { p.update(t); p.draw(); }

      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < CONNECT_DIST) {
            const lineAlpha = (1 - dist / CONNECT_DIST) * (isDark ? 0.12 : 0.08);
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.strokeStyle = colors[particles[i].colorIdx] + lineAlpha + ")";
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        }
      }

      for (const p of particles) {
        const dx = p.x - mouse.x;
        const dy = p.y - mouse.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < MOUSE_DIST) {
          const lineAlpha = (1 - dist / MOUSE_DIST) * (isDark ? 0.25 : 0.15);
          ctx.beginPath();
          ctx.moveTo(p.x, p.y);
          ctx.lineTo(mouse.x, mouse.y);
          ctx.strokeStyle = `rgba(0,201,167,${lineAlpha})`;
          ctx.lineWidth = 0.8;
          ctx.stroke();
        }
      }

      animId = requestAnimationFrame(animate);
    };
    animate();

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("resize", resize);
      window.removeEventListener("mousemove", handleMouse);
    };
  }, [isDark]);

  return (
    <canvas ref={canvasRef} style={{
      position: "fixed", top: 0, left: 0, width: "100%", height: "100%",
      zIndex: 0, pointerEvents: "none",
    }} />
  );
}
// ════════════════════════════════════════════════════════════
// HOME PAGE
// ════════════════════════════════════════════════════════════
function HomePage({ t, metrics, setPage, cardStyle, btnPrimary, btnOutline, gallery, selectGallery }) {
  const TIERS = [
    {
      name: "Basic", price: "Free", sub: "For researchers & students",
      features: ["5 analyses per day", "ECF-DPO mode only", "Sample gallery access", "Community support"],
      cta: "Try Free", highlight: false,
    },
    {
      name: "Professional", price: "$49/month", sub: "For clinicians & labs",
      features: ["Unlimited analyses", "ECF-DPO with Verifier mode", "Custom image upload", "Report generation", "Priority support"],
      cta: "Start Trial", highlight: true,
    },
    {
      name: "Enterprise", price: "$499/year", sub: "For hospitals & health systems",
      features: ["On-premise deployment", "HIPAA compliant", "Custom model fine-tuning", "API integration", "Dedicated support", "SLA guarantee"],
      cta: "Contact Sales", highlight: false,
    },
  ];

  return (
    <div className="fade-in">
      {/* Hero */}
      <section style={{ textAlign: "center", padding: "4rem 0 2.5rem" }}>
        <div style={{
          display: "inline-block", fontSize: 10, fontWeight: 600, letterSpacing: 1.5,
          textTransform: "uppercase", color: t.accent, background: `${t.accent}15`,
          padding: "4px 14px", borderRadius: 20, marginBottom: 16,
          fontFamily: "'Fira Code',monospace",
        }}>ECF-DPO Peer-Reviewed Research</div>
        <h1 style={{
          fontFamily: "'DM Serif Display',serif", fontSize: "clamp(2.5rem,6vw,4rem)",
          color: t.textHeading, lineHeight: 1.1, marginBottom: 12,
        }}>
          Endo<span style={{ color: t.accent }}>Guard</span>
        </h1>
        <p style={{ fontSize: 18, color: t.textMuted, maxWidth: 600, margin: "0 auto 8px" }}>
          Reducing AI Hallucinations in GI Endoscopy
        </p>
        <p style={{ fontSize: 14, color: t.textMuted, marginBottom: 28 }}>
          A Trustworthy Medical VQA Framework for Clinical Endoscopy Analysis
        </p>
        <button onClick={() => setPage("analyze")} style={{ ...btnPrimary, padding: "0.8rem 2rem", fontSize: 15 }}>
          Try EndoGuard Live <Icon name="arrow" size={16} />
        </button>
      </section>

      {/* 3 Metrics cards */}
      <section style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 16, marginBottom: 48 }}>
        {[
          { label: "Refusal Accuracy", from: "11.9%", to: "60.7%", delta: "+410%", color: "#22c55e" },
          { label: "Grounded Semantic Score", from: "24.8%", to: "51.3%", delta: "+107%", color: "#3b82f6" },
          { label: "Hallucination Rate (CHAIRs)", from: "43.1%", to: "35.7%", delta: "↓17%", color: "#f97316" },
        ].map((m, i) => (
          <div key={i} className="hover-lift" style={{ ...cardStyle, textAlign: "center" }}>
            <div style={{ fontSize: 11, color: t.textMuted, fontWeight: 500, marginBottom: 8 }}>{m.label}</div>
            <div style={{ fontSize: 12, color: t.textMuted, marginBottom: 2 }}>{m.from} →</div>
            <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 28, fontWeight: 700, color: t.textHeading }}>{m.to}</div>
            <div style={{ fontSize: 12, fontWeight: 600, color: m.color, marginTop: 4 }}>{m.delta}</div>
          </div>
        ))}
      </section>

      {/* Sample Gallery */}
      <section style={{ marginBottom: 48 }}>
        <div style={{ textAlign: "center", marginBottom: 20 }}>
          <div style={{ fontSize: 10, color: t.textMuted, fontFamily: "'Fira Code',monospace", letterSpacing: "0.2em", textTransform: "uppercase", marginBottom: 8 }}>Test With Real Cases</div>
          <h2 style={{ fontFamily: "'DM Serif Display',serif", fontSize: 26, color: t.textHeading }}>Sample Endoscopy Gallery</h2>
          <p style={{ fontSize: 13, color: t.textMuted, marginTop: 6, maxWidth: 520, margin: "6px auto 0" }}>
            Pre-loaded clinical images covering abnormalities, landmarks, and instruments. Tap any image to run EndoGuard analysis on a known case.
          </p>
        </div>

        {/* Category legend */}
        {gallery.length > 0 && (
          <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "center", gap: 8, marginBottom: 16 }}>
            {[...new Map(gallery.map(g => [g.category, g.color || t.accent])).entries()].map(([cat, color]) => (
              <span key={cat} style={{
                fontSize: 10, fontWeight: 600, padding: "3px 10px", borderRadius: 20,
                background: `${color}15`, color: color, border: `1px solid ${color}30`,
                fontFamily: "'Fira Code',monospace",
              }}>{cat}</span>
            ))}
          </div>
        )}

        {gallery.length > 0 ? (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(200px,1fr))", gap: 14 }}>
            {gallery.map((g) => {
              const catColor = g.color || t.accent;
              return (
                <div key={g.id} className="hover-lift" onClick={() => { selectGallery(g); setPage("analyze"); }}
                  style={{
                    ...cardStyle, padding: 0, overflow: "hidden", cursor: "pointer",
                    border: `1px solid ${t.border}`, transition: "all 0.25s",
                  }}>
                  {/* Image thumbnail */}
                  <div style={{ position: "relative", background: t.bgCardSolid, minHeight: 140 }}>
                    {g.thumb ? (
                      <img src={g.thumb} alt={g.category}
                        style={{ width: "100%", height: 140, objectFit: "cover", display: "block" }} />
                    ) : (
                      <div style={{ width: "100%", height: 140, display: "flex", alignItems: "center", justifyContent: "center", color: t.textMuted, fontSize: 12 }}>
                        Loading...
                      </div>
                    )}
                    {/* Category badge on image */}
                    <span style={{
                      position: "absolute", top: 6, right: 6, fontSize: 10, fontWeight: 700,
                      padding: "3px 10px", borderRadius: 5,
                      background: `${catColor}22`, color: catColor, border: `1px solid ${catColor}50`,
                      fontFamily: "'Fira Code',monospace", backdropFilter: "blur(8px)",
                    }}>{g.category}</span>
                  </div>
                  {/* File info bar */}
                  <div style={{ padding: "8px 10px", borderTop: `1px solid ${t.border}` }}>
                    <div style={{ fontSize: 11, color: t.text, fontFamily: "'Fira Code',monospace", fontWeight: 500, marginBottom: 2, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {g.id}.jpg
                    </div>
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                      <span style={{ fontSize: 10, color: catColor, fontWeight: 600 }}>{g.category}</span>
                      <span style={{ fontSize: 11, color: t.accent, fontWeight: 600 }}>Analyze →</span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div style={{ ...cardStyle, textAlign: "center", padding: "2.5rem", color: t.textMuted }}>
            <Icon name="upload" size={32} color={t.textMuted} />
            <p style={{ marginTop: 10, fontSize: 14 }}>Connect the API to load sample endoscopy images</p>
            <p style={{ fontSize: 12, marginTop: 4 }}>Gallery images are served from the Kvasir-VQA v2 dataset on Kaggle</p>
          </div>
        )}
        {gallery.length > 0 && (
          <div style={{ textAlign: "center", marginTop: 16 }}>
            <button onClick={() => setPage("analyze")} style={btnOutline}>
              View all samples & upload your own <Icon name="arrow" size={14} />
            </button>
          </div>
        )}
      </section>

      {/* Pricing */}
      <section style={{ marginBottom: 48 }}>
        <div style={{ textAlign: "center", marginBottom: 28 }}>
          <div style={{ fontSize: 10, color: t.textMuted, fontFamily: "'Fira Code',monospace", letterSpacing: "0.2em", textTransform: "uppercase", marginBottom: 8 }}>Plans</div>
          <h2 style={{ fontFamily: "'DM Serif Display',serif", fontSize: 26, color: t.textHeading }}>Deployment Options</h2>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 14 }}>
          {TIERS.map((p, i) => (
            <div key={i} className="hover-lift" style={{
              ...cardStyle, padding: "32px 24px", position: "relative",
              border: p.highlight ? `1.5px solid ${t.accent}` : `1px solid ${t.border}`,
            }}>
              {p.highlight && <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, borderRadius: "14px 14px 0 0", background: `linear-gradient(90deg,${t.accent},${t.accentAlt})` }} />}
              <div style={{ fontSize: 13, fontWeight: 700, color: t.accent, fontFamily: "'Fira Code',monospace", letterSpacing: "0.06em" }}>{p.name}</div>
              <div style={{ fontFamily: "'DM Serif Display',serif", fontSize: 32, color: t.textHeading, marginTop: 8 }}>{p.price}</div>
              <div style={{ fontSize: 12, color: t.textMuted, marginBottom: 20 }}>{p.sub}</div>
              {p.features.map((f, j) => (
                <div key={j} style={{ fontSize: 13, color: t.text, marginBottom: 8, paddingLeft: 16, position: "relative", lineHeight: 1.6 }}>
                  <span style={{ position: "absolute", left: 0, color: t.accent }}>✓</span>{f}
                </div>
              ))}
              <button onClick={() => p.name === "Basic" && setPage("analyze")} style={{
                width: "100%", marginTop: 20, padding: "10px", fontSize: 13, fontWeight: 600,
                background: p.highlight ? `linear-gradient(135deg,${t.accent},${t.accentAlt})` : "transparent",
                color: p.highlight ? "#fff" : t.accent,
                border: p.highlight ? "none" : `1px solid ${t.accent}30`,
                borderRadius: 10, cursor: "pointer", fontFamily: "'DM Sans',sans-serif",
                transition: "all 0.2s",
              }}>{p.cta}</button>
            </div>
          ))}
        </div>
      </section>

      {/* Tech footer */}
      <section style={{ textAlign: "center", padding: "2rem 0", borderTop: `1px solid ${t.border}` }}>
        <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: 1.5, color: t.textMuted, marginBottom: 12 }}>Technology Stack</div>
        <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "center", gap: 8 }}>
          {["MedGemma 4B", "Kvasir-VQA v2", "3-Stage DPO", "BiomedCLIP", "94MB Adapter", "FastAPI"].map(t2 => (
            <span key={t2} style={{
              fontFamily: "'Fira Code',monospace", fontSize: 11, padding: "4px 12px",
              borderRadius: 6, background: `${t.accent}10`, border: `1px solid ${t.border}`, color: t.textMuted,
            }}>{t2}</span>
          ))}
        </div>
      </section>
    </div>
  );
}

// ════════════════════════════════════════════════════════════
// ANALYZE PAGE
// ════════════════════════════════════════════════════════════
function AnalyzePage({
  t, mode, setMode, image, preview, selectedGalleryId, gallery, result,
  loading, elapsed, qa, qaInput, qaLoading, showReport, fileRef,
  cardStyle, btnPrimary, btnOutline, handleFile, selectGallery,
  runAnalysis, askFollowUp, setQaInput, setShowReport, resetAnalysis
}) {
  const hasImage = !!(image || selectedGalleryId);

  // ── Report modal ──
  if (showReport && result) {
    return (
      <ReportView t={t} result={result} preview={preview} mode={mode}
        onClose={() => setShowReport(false)} />
    );
  }

  return (
    <div className="fade-in" style={{ paddingTop: 24 }}>
      {/* Mode toggle + New Analysis button */}
      <div className="no-print" style={{ display: "flex", justifyContent: "center", alignItems: "center", gap: 12, marginBottom: 24, flexWrap: "wrap" }}>
        <div style={{
          display: "inline-flex", background: t.bgCard, border: `1px solid ${t.border}`,
          borderRadius: 12, padding: 4, backdropFilter: "blur(12px)",
        }}>
          {[
            { id: "dpo_only", label: "ECF-DPO Only", c: "#00c9a7" },
            { id: "dpo_with_verifier", label: "ECF-DPO with Verifier", c: "#38bdf8", icon: true },
          ].map(m => (
            <button key={m.id} onClick={() => setMode(m.id)} style={{
              background: mode === m.id ? `${m.c}20` : "transparent",
              color: mode === m.id ? m.c : t.textMuted,
              border: mode === m.id ? `1px solid ${m.c}40` : "1px solid transparent",
              borderRadius: 8, padding: "8px 20px", cursor: "pointer",
              fontFamily: "'DM Sans'", fontWeight: 600, fontSize: 13,
              display: "flex", alignItems: "center", gap: 6, transition: "all 0.2s",
            }}>
              {m.icon && <Icon name="shield" size={14} />}
              {m.label}
            </button>
          ))}
        </div>
        {/* New Analysis button — shown when an image is loaded or results exist */}
        {(hasImage || result) && (
          <button onClick={resetAnalysis} style={{
            ...btnOutline, display: "flex", alignItems: "center", gap: 6,
            borderColor: "#f97316", color: "#f97316",
          }}>
            ✕ New Analysis
          </button>
        )}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "minmax(0,1fr) minmax(0,1.2fr)", gap: 20, alignItems: "start" }}>
        {/* Left column */}
        <div>
          {/* Upload zone */}
          <div style={{
            ...cardStyle, marginBottom: 16, textAlign: "center",
            border: `2px dashed ${preview ? t.accent : t.border}`,
            cursor: "pointer", position: "relative", overflow: "hidden", minHeight: 200,
          }}
            onClick={() => fileRef.current?.click()}
            onDragOver={e => e.preventDefault()}
            onDrop={e => { e.preventDefault(); handleFile(e.dataTransfer.files[0]); }}
          >
            <input ref={fileRef} type="file" accept="image/*" style={{ display: "none" }}
              onChange={e => handleFile(e.target.files[0])} />
            {preview ? (
              <img src={preview} alt="Selected" style={{
                maxWidth: "100%", maxHeight: 320, borderRadius: 8, objectFit: "contain",
              }} />
            ) : (
              <div style={{ padding: "2.5rem 1rem" }}>
                <Icon name="upload" size={40} color={t.accent} />
                <p style={{ color: t.textHeading, marginTop: 12, fontSize: 15, fontWeight: 600 }}>
                  Upload Endoscopy Image
                </p>
                <p style={{ color: t.textMuted, marginTop: 6, fontSize: 12, lineHeight: 1.5 }}>
                  Drag & drop a .jpg or .png file here, or click to browse.
                  <br />Or select a sample from the gallery below.
                </p>
              </div>
            )}
          </div>

          {/* Gallery thumbnails */}
          {gallery.length > 0 && (
            <div style={cardStyle}>
              <div style={{ fontSize: 12, fontWeight: 600, color: t.textMuted, marginBottom: 8 }}>Sample Gallery — tap to analyze</div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                {gallery.map(g => (
                  <div key={g.id} onClick={() => selectGallery(g)} style={{
                    cursor: "pointer", borderRadius: 8, overflow: "hidden",
                    border: selectedGalleryId === g.id ? `2px solid ${t.accent}` : `1px solid ${t.border}`,
                    width: 80, transition: "all 0.2s", position: "relative",
                  }}>
                    {g.thumb ? (
                      <img src={g.thumb} alt={g.category}
                        style={{ width: "100%", height: 58, objectFit: "cover", display: "block" }} />
                    ) : (
                      <div style={{ width: "100%", height: 58, background: t.bgCardSolid }} />
                    )}
                    <div style={{
                      fontSize: 7, textAlign: "center", padding: "2px 1px",
                      background: t.bgCardSolid, color: g.color || t.textMuted,
                      fontWeight: 700, lineHeight: 1.2,
                    }}>{g.category}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Analyze button */}
          <button onClick={() => runAnalysis()} disabled={!hasImage || loading}
            style={{
              ...btnPrimary, width: "100%", marginTop: 16, padding: "0.8rem",
              opacity: (!hasImage || loading) ? 0.5 : 1,
              display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
            }}>
            {loading ? <><Icon name="loader" size={16} /> Analyzing... {elapsed.toFixed(1)}s</> : "Analyze Image"}
          </button>
        </div>

        {/* Right column — Results */}
        <div>
          {loading && (
            <div style={{ ...cardStyle, textAlign: "center", padding: "3rem" }}>
              <Icon name="loader" size={32} color={t.accent} />
              <p style={{ color: t.textMuted, marginTop: 12 }}>Running {mode === "dpo_with_verifier" ? "DPO + Verifier" : "DPO"} analysis...</p>
              <p style={{ fontFamily: "'Fira Code',monospace", color: t.accent, fontSize: 20, marginTop: 8 }}>{elapsed.toFixed(1)}s</p>
            </div>
          )}

          {result?.error && (
            <div style={{ ...cardStyle, background: "#ef444415", borderColor: "#ef444440" }}>
              <p style={{ color: "#ef4444" }}>Error: {result.error}</p>
            </div>
          )}

          {result && !result.error && (
            <div className="fade-in">
              {/* Entity chips */}
              <div style={{ ...cardStyle, marginBottom: 12 }}>
                <div style={{ fontSize: 11, fontWeight: 600, color: t.textMuted, marginBottom: 8 }}>Detected Entities</div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                  {(result.entities?.highlighted || []).map((e, i) => {
                    const statusIcon = e.status === "verified" ? "check" : e.status === "flagged" ? "alert" : e.status === "unchecked" ? "help" : null;
                    return (
                      <span key={i} style={{
                        display: "inline-flex", alignItems: "center", gap: 4,
                        padding: "3px 10px", borderRadius: 6, fontSize: 12, fontWeight: 600,
                        background: `${e.color}18`, color: e.color, border: `1px solid ${e.color}30`,
                      }}>
                        {statusIcon && <Icon name={statusIcon} size={12} color={e.color} />}
                        {e.entity}
                      </span>
                    );
                  })}
                  {(result.entities?.highlighted || []).length === 0 && (
                    <span style={{ fontSize: 12, color: t.textMuted }}>No entities detected</span>
                  )}
                </div>
              </div>

              {/* Response(s) */}
              {result.mode === "dpo_only" ? (
                <div style={cardStyle}>
                  <div style={{ fontSize: 11, fontWeight: 600, color: t.textMuted, marginBottom: 8 }}>Model Response</div>
                  <p style={{ fontSize: 14, lineHeight: 1.7, color: t.text }}
                    dangerouslySetInnerHTML={{ __html: highlightEntities(result.response, result.entities?.mentioned || []) }} />
                  <div style={{ fontSize: 11, color: t.textMuted, marginTop: 8 }}>
                    ⏱ {result.inference_time} · 📐 {result.image_size}
                  </div>
                </div>
              ) : (
                <>
                  {/* Initial response */}
                  <div style={{ ...cardStyle, marginBottom: 12, borderLeft: `3px solid #f97316` }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                      <div style={{ fontSize: 11, fontWeight: 600, color: "#f97316" }}>Initial Response</div>
                      {result.was_corrected && (
                        <span style={{
                          fontSize: 10, fontWeight: 600, padding: "2px 8px", borderRadius: 4,
                          background: "#ef444415", color: "#ef4444", fontFamily: "'Fira Code',monospace",
                        }}>{result.correction_type?.toUpperCase()}</span>
                      )}
                    </div>
                    <p style={{ fontSize: 14, lineHeight: 1.7, color: t.text }}
                      dangerouslySetInnerHTML={{
                        __html: highlightEntities(
                          result.initial_response,
                          result.entities?.mentioned || [],
                          result.entities?.flagged || [],
                        )
                      }} />
                  </div>
                  {/* Final response */}
                  <div style={{ ...cardStyle, marginBottom: 12, borderLeft: `3px solid #22c55e` }}>
                    <div style={{ fontSize: 11, fontWeight: 600, color: "#22c55e", marginBottom: 8 }}>Final Response (Verified)</div>
                    <p style={{ fontSize: 14, lineHeight: 1.7, color: t.text }}
                      dangerouslySetInnerHTML={{
                        __html: highlightEntities(
                          result.final_response,
                          [...(result.entities?.verified || []), ...(result.entities?.unchecked || [])],
                        )
                      }} />
                    <div style={{ fontSize: 11, color: t.textMuted, marginTop: 8 }}>⏱ {result.inference_time} · 📐 {result.image_size}</div>
                  </div>

                  {/* Verifier confidence */}
                  {result.verifier && (
                    <div style={cardStyle}>
                      <div style={{ fontSize: 11, fontWeight: 600, color: t.textMuted, marginBottom: 12 }}>
                        <Icon name="shield" size={13} color={t.accentAlt} /> Classifier Confidence (threshold: {result.verifier.threshold})
                      </div>
                      {Object.entries(result.verifier.classifier_scores || {}).map(([ent, score]) => {
                        const isAbove = score >= result.verifier.threshold;
                        const barColor = isAbove ? "#22c55e" : "#ef4444";
                        return (
                          <div key={ent} style={{ marginBottom: 10 }}>
                            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 3 }}>
                              <span style={{ color: t.text, fontWeight: 500, textTransform: "capitalize" }}>{ent}</span>
                              <span style={{ fontFamily: "'Fira Code',monospace", color: barColor, fontWeight: 600 }}>{score.toFixed(3)}</span>
                            </div>
                            <div style={{ position: "relative", height: 10, background: `${t.border}`, borderRadius: 5 }}>
                              <div style={{
                                height: "100%", borderRadius: 5, width: `${Math.min(score * 100, 100)}%`,
                                background: barColor, transition: "width 0.5s ease",
                              }} />
                              <div style={{
                                position: "absolute", top: -3, bottom: -3,
                                left: `${result.verifier.threshold * 100}%`,
                                width: 1, borderLeft: `2px dashed ${t.textMuted}`,
                              }} />
                            </div>
                          </div>
                        );
                      })}
                      <div style={{ fontSize: 10, color: t.textMuted, marginTop: 4 }}>
                        Verification precision: {result.verifier.precision}
                      </div>
                    </div>
                  )}
                </>
              )}

              {/* Actions */}
              <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
                <button onClick={() => setShowReport(true)} style={btnOutline}>
                  <Icon name="file" size={14} /> Generate Report
                </button>
              </div>

              {/* Follow-up Q&A */}
              <div style={{ ...cardStyle, marginTop: 16 }}>
                <div style={{ fontSize: 11, fontWeight: 600, color: t.textMuted, marginBottom: 8 }}>Follow-up Questions</div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 12 }}>
                  {SUGGESTED_QS.map((q, i) => (
                    <button key={i} onClick={() => askFollowUp(q)} disabled={qaLoading}
                      style={{
                        fontSize: 11, padding: "4px 10px", borderRadius: 6, cursor: "pointer",
                        background: `${t.accent}10`, border: `1px solid ${t.border}`, color: t.textMuted,
                        fontFamily: "'DM Sans'", transition: "all 0.2s",
                      }}>{q}</button>
                  ))}
                </div>
                <div style={{ display: "flex", gap: 8 }}>
                  <input value={qaInput} onChange={e => setQaInput(e.target.value)}
                    onKeyDown={e => e.key === "Enter" && askFollowUp(qaInput)}
                    placeholder="Ask a follow-up question..."
                    style={{
                      flex: 1, background: t.inputBg, border: `1px solid ${t.border}`, borderRadius: 8,
                      padding: "8px 12px", color: t.text, fontSize: 13, fontFamily: "'DM Sans'", outline: "none",
                    }} />
                  <button onClick={() => askFollowUp(qaInput)} disabled={qaLoading || !qaInput.trim()}
                    style={{ ...btnPrimary, padding: "8px 14px", opacity: qaLoading || !qaInput.trim() ? 0.5 : 1 }}>
                    {qaLoading ? <Icon name="loader" size={14} /> : <Icon name="send" size={14} />}
                  </button>
                </div>
                {qa.map((item, i) => (
                  <div key={i} style={{ marginTop: 12, padding: "10px 12px", background: `${t.accent}06`, borderRadius: 8, border: `1px solid ${t.border}` }}>
                    <div style={{ fontSize: 12, fontWeight: 600, color: t.accent, marginBottom: 4 }}>Q: {item.q}</div>
                    <div style={{ fontSize: 13, color: t.text, lineHeight: 1.6 }}
                      dangerouslySetInnerHTML={{ __html: highlightEntities(item.a, Object.keys(ENTITY_COLORS)) }} />
                  </div>
                ))}
              </div>
            </div>
          )}

          {!result && !loading && (
            <div style={{ ...cardStyle, textAlign: "center", padding: "3rem", color: t.textMuted }}>
              <Icon name="search" size={36} color={t.textMuted} />
              <p style={{ marginTop: 12, fontSize: 14 }}>Select or upload an image, then click Analyze</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════════════════
// REPORT VIEW
// ════════════════════════════════════════════════════════════
function ReportView({ t, result, preview, mode, onClose }) {
  return (
    <div style={{ paddingTop: 20, maxWidth: 800, margin: "0 auto" }}>
      <div className="no-print" style={{ display: "flex", justifyContent: "space-between", marginBottom: 16 }}>
        <button onClick={onClose} style={{ background: "transparent", border: "none", color: t.accent, cursor: "pointer", fontFamily: "'DM Sans'", fontSize: 14 }}>
          ← Back to analysis
        </button>
        <button onClick={() => window.print()} style={{
          background: `linear-gradient(135deg, ${t.accent}, ${t.accentAlt})`, color: "#fff",
          border: "none", borderRadius: 8, padding: "6px 16px", cursor: "pointer", fontFamily: "'DM Sans'", fontWeight: 600, fontSize: 13,
          display: "flex", alignItems: "center", gap: 6,
        }}>
          <Icon name="download" size={14} /> Download PDF
        </button>
      </div>

      <div style={{ background: t.bgCardSolid, border: `1px solid ${t.border}`, borderRadius: 12, padding: "2rem", lineHeight: 1.7 }}>
        <div style={{ textAlign: "center", marginBottom: 24 }}>
          <h1 style={{ fontFamily: "'DM Serif Display',serif", fontSize: 28, color: t.textHeading }}>
            EndoGuard Analysis Report
          </h1>
          <p style={{ fontSize: 12, color: t.textMuted }}>{new Date().toLocaleString()}</p>
          <span style={{
            display: "inline-block", fontSize: 10, fontWeight: 600, padding: "2px 10px",
            borderRadius: 4, background: `${t.accent}15`, color: t.accent, marginTop: 6,
            fontFamily: "'Fira Code',monospace",
          }}>ECF-DPO {mode === "dpo_with_verifier" ? " WITH VERIFIER" : " ONLY"}</span>
        </div>

        {preview && (
          <div style={{ textAlign: "center", marginBottom: 20 }}>
            <img src={preview} alt="Analyzed" style={{ maxWidth: 300, borderRadius: 8, border: `1px solid ${t.border}` }} />
          </div>
        )}

        <h3 style={{ fontFamily: "'DM Serif Display',serif", fontSize: 18, color: t.textHeading, marginBottom: 8 }}>Question</h3>
        <p style={{ fontSize: 14, color: t.text, marginBottom: 16 }}>{result.question}</p>

        <h3 style={{ fontFamily: "'DM Serif Display',serif", fontSize: 18, color: t.textHeading, marginBottom: 8 }}>Entities Detected</h3>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 16 }}>
          {(result.entities?.highlighted || []).map((e, i) => (
            <span key={i} style={{
              padding: "3px 10px", borderRadius: 5, fontSize: 12, fontWeight: 600,
              background: `${e.color}18`, color: e.color,
            }}>{e.entity} {e.status ? `(${e.status})` : ""}</span>
          ))}
        </div>

        <h3 style={{ fontFamily: "'DM Serif Display',serif", fontSize: 18, color: t.textHeading, marginBottom: 8 }}>Response</h3>
        <p style={{ fontSize: 14, color: t.text, marginBottom: 16 }}>
          {result.response || result.final_response}
        </p>

        {result.verifier && (
          <>
            <h3 style={{ fontFamily: "'DM Serif Display',serif", fontSize: 18, color: t.textHeading, marginBottom: 8 }}>Verifier Scores</h3>
            <table style={{ width: "100%", fontSize: 13, borderCollapse: "collapse", marginBottom: 16 }}>
              <thead><tr>
                {["Entity", "Score", "Status"].map(h => (
                  <th key={h} style={{ textAlign: "left", padding: "6px 10px", borderBottom: `1px solid ${t.border}`, fontWeight: 600, color: t.textMuted }}>{h}</th>
                ))}
              </tr></thead>
              <tbody>
                {Object.entries(result.verifier.classifier_scores).map(([ent, score]) => (
                  <tr key={ent}>
                    <td style={{ padding: "6px 10px", textTransform: "capitalize", borderBottom: `1px solid ${t.border}` }}>{ent}</td>
                    <td style={{ padding: "6px 10px", fontFamily: "'Fira Code',monospace", borderBottom: `1px solid ${t.border}` }}>{score.toFixed(3)}</td>
                    <td style={{ padding: "6px 10px", borderBottom: `1px solid ${t.border}`, color: score >= 0.5 ? "#22c55e" : "#ef4444", fontWeight: 600 }}>
                      {score >= 0.5 ? "Verified" : "Flagged"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}

        <div style={{ fontSize: 11, color: t.textMuted, textAlign: "center", marginTop: 20, paddingTop: 12, borderTop: `1px solid ${t.border}` }}>
          Generated by EndoGuard · Entity Consistency Framework · {result.inference_time}
        </div>
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════════════════
// METHODOLOGY PAGE
// ════════════════════════════════════════════════════════════
function MethodologyPage({ t, cardStyle, expandedPhase, setExpandedPhase, metrics }) {
  const phases = [
    {
      id: "data", n: "01", title: "Data Preparation", color: "#a855f7",
      summary: "Build clinical confusability table and generate three types of negatives for DPO training.",
      details: [
        "Kvasir-VQA v2 dataset provides endoscopy image–question–answer triplets.",
        "Confusability table maps visually similar entities (e.g. polyp ↔ ulcerative colitis).",
        "Type 1 — Entity Swap: replace correct entity with confusable alternative.",
        "Type 2 — Image Mismatch: pair answer with wrong image from same category.",
        "Type 3 — Multi-Entity: fabricate extra entity mentions not present in image.",
        "Negatives split into Easy tier (obvious errors) and Hard tier (subtle confusions).",
      ],
    },
    {
      id: "dpo", n: "02", title: "Curriculum DPO Training", color: "#00c9a7",
      summary: "Progressive fine-tuning from easy to hard negatives using Direct Preference Optimization.",
      details: [
        "Stage 1: 2 epochs on easy negatives — model learns to avoid obvious hallucinations.",
        "Stage 2: 2 epochs on hard negatives — model learns subtle entity distinctions.",
        "Stage 3: 1 epoch on mixed negatives — consolidation and generalization.",
        "Weights carry forward between stages (curriculum learning).",
        "Base model: MedGemma-4B-IT with QLoRA (rank=16, 4-bit NF4 quantization).",
        "DPO objective with β=0.1, total training time ~15.1 hours on T4 GPU.",
      ],
    },
    {
      id: "verify", n: "03", title: "Entity Verification", color: "#38bdf8",
      summary: "Post-hoc cross-checking of entity claims using a trained BiomedCLIP classifier.",
      details: [
        "Model output → Entity Extractor parses mentioned entities from response.",
        "Negation detection identifies denied entities (e.g. 'no polyp visible').",
        "BiomedCLIP vision encoder extracts image features (frozen, 512-dim).",
        "Trained MLP head classifies 3 entities: polyp, UC, oesophagitis (macro F1: 0.944).",
        "Cross-check: if model claims entity but classifier doesn't see it → flagged.",
        "Hybrid correction: header-only edit for denied entities, model rewrite for genuine claims.",
      ],
    },
  ];

  return (
    <div className="fade-in" style={{ paddingTop: 24 }}>
      <h1 style={{
        fontFamily: "'DM Serif Display',serif", fontSize: "clamp(1.8rem,4vw,2.5rem)",
        color: t.textHeading, textAlign: "center", marginBottom: 8,
      }}>Methodology</h1>
      <p style={{ textAlign: "center", color: t.textMuted, fontSize: 14, marginBottom: 32, maxWidth: 600, margin: "0 auto 32px" }}>
        EndoGuard's three-phase pipeline for reducing hallucinations in medical VQA
      </p>

      {/* Pipeline flow */}
      <div style={{ display: "flex", flexDirection: "column", gap: 0, marginBottom: 40 }}>
        {phases.map((p, i) => (
          <div key={p.id}>
            <div className="hover-lift" onClick={() => setExpandedPhase(expandedPhase === p.id ? null : p.id)}
              style={{
                ...cardStyle, cursor: "pointer", borderLeft: `4px solid ${p.color}`,
                marginBottom: i < 2 ? 0 : undefined,
              }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                  <span style={{
                    fontFamily: "'Fira Code',monospace", fontSize: 11, fontWeight: 700,
                    color: p.color, background: `${p.color}15`, padding: "4px 10px", borderRadius: 6,
                  }}>PHASE {p.n}</span>
                  <h3 style={{ fontFamily: "'DM Serif Display',serif", fontSize: 20, color: t.textHeading }}>{p.title}</h3>
                </div>
                <span style={{ color: t.textMuted, fontSize: 18, transition: "transform 0.2s", transform: expandedPhase === p.id ? "rotate(90deg)" : "none" }}>›</span>
              </div>
              <p style={{ fontSize: 13, color: t.textMuted, marginTop: 8, lineHeight: 1.5 }}>{p.summary}</p>
              {expandedPhase === p.id && (
                <div className="fade-in" style={{ marginTop: 12, paddingTop: 12, borderTop: `1px solid ${t.border}` }}>
                  {p.details.map((d, j) => (
                    <div key={j} style={{ display: "flex", gap: 8, marginBottom: 6, fontSize: 13, color: t.text, lineHeight: 1.6 }}>
                      <span style={{ color: p.color, fontWeight: 700, flexShrink: 0 }}>•</span>
                      <span>{d}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
            {i < 2 && (
              <div style={{ display: "flex", justifyContent: "center", padding: "4px 0" }}>
                <svg width="20" height="24" viewBox="0 0 20 24"><path d="M10 0v20M4 16l6 6 6-6" stroke={t.textMuted} strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round" /></svg>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Pipeline diagram */}
      <div style={{ ...cardStyle, marginBottom: 32, overflowX: "auto" }}>
        <h3 style={{ fontFamily: "'DM Serif Display',serif", fontSize: 18, color: t.textHeading, marginBottom: 16 }}>System Architecture</h3>
        <svg viewBox="0 0 900 220" style={{ width: "100%", maxWidth: 900 }}>
          {/* Phase 1 */}
          <rect x="10" y="20" width="260" height="180" rx="10" fill={t.bgCardSolid} stroke="#a855f7" strokeWidth="1.5" strokeDasharray="4" />
          <text x="140" y="14" textAnchor="middle" fill="#a855f7" fontSize="10" fontWeight="700" fontFamily="'Fira Code',monospace">PHASE 1: DATA PREP</text>
          <rect x="30" y="40" width="100" height="30" rx="6" fill="#a855f720" stroke="#a855f7" strokeWidth="1" />
          <text x="80" y="59" textAnchor="middle" fill={t.text} fontSize="9" fontFamily="'DM Sans',sans-serif">Kvasir-VQA</text>
          <rect x="150" y="40" width="100" height="30" rx="6" fill="#a855f720" stroke="#a855f7" strokeWidth="1" />
          <text x="200" y="59" textAnchor="middle" fill={t.text} fontSize="9" fontFamily="'DM Sans',sans-serif">Confusability</text>
          <line x1="130" y1="55" x2="150" y2="55" stroke={t.textMuted} strokeWidth="1" markerEnd="url(#arrow)" />
          {["T1: Swap", "T2: Mismatch", "T3: Multi"].map((label, i) => (
            <g key={i}>
              <rect x={30 + i * 82} y="90" width="75" height="26" rx="5" fill="#a855f710" stroke="#a855f740" strokeWidth="1" />
              <text x={30 + i * 82 + 37} y="107" textAnchor="middle" fill={t.textMuted} fontSize="8" fontFamily="'Fira Code',monospace">{label}</text>
            </g>
          ))}
          <rect x="55" y="135" width="80" height="24" rx="5" fill="#22c55e15" stroke="#22c55e" strokeWidth="1" />
          <text x="95" y="151" textAnchor="middle" fill="#22c55e" fontSize="8" fontFamily="'Fira Code',monospace">Easy Tier</text>
          <rect x="155" y="135" width="80" height="24" rx="5" fill="#ef444415" stroke="#ef4444" strokeWidth="1" />
          <text x="195" y="151" textAnchor="middle" fill="#ef4444" fontSize="8" fontFamily="'Fira Code',monospace">Hard Tier</text>

          {/* Arrow 1→2 */}
          <line x1="270" y1="110" x2="310" y2="110" stroke={t.textMuted} strokeWidth="1.5" markerEnd="url(#arrowM)" />

          {/* Phase 2 */}
          <rect x="310" y="20" width="260" height="180" rx="10" fill={t.bgCardSolid} stroke="#00c9a7" strokeWidth="1.5" strokeDasharray="4" />
          <text x="440" y="14" textAnchor="middle" fill="#00c9a7" fontSize="10" fontWeight="700" fontFamily="'Fira Code',monospace">PHASE 2: CURRICULUM DPO</text>
          {["S1: Easy (2ep)", "S2: Hard (2ep)", "S3: Mixed (1ep)"].map((label, i) => (
            <g key={i}>
              <rect x={330} y={45 + i * 45} width="120" height="30" rx="6" fill="#00c9a720" stroke="#00c9a7" strokeWidth="1" />
              <text x={390} y={64 + i * 45} textAnchor="middle" fill={t.text} fontSize="9" fontFamily="'DM Sans',sans-serif">{label}</text>
              {i < 2 && <line x1="390" y1={75 + i * 45} x2="390" y2={90 + i * 45} stroke="#00c9a7" strokeWidth="1" markerEnd="url(#arrowG)" />}
            </g>
          ))}
          <rect x="470" y="75" width="85" height="40" rx="6" fill="#00c9a710" stroke="#00c9a740" strokeWidth="1" />
          <text x="512" y="92" textAnchor="middle" fill={t.textMuted} fontSize="8" fontFamily="'Fira Code',monospace">MedGemma-4B</text>
          <text x="512" y="105" textAnchor="middle" fill={t.textMuted} fontSize="8" fontFamily="'Fira Code',monospace">+ QLoRA</text>

          {/* Arrow 2→3 */}
          <line x1="570" y1="110" x2="610" y2="110" stroke={t.textMuted} strokeWidth="1.5" markerEnd="url(#arrowM)" />

          {/* Phase 3 */}
          <rect x="610" y="20" width="280" height="180" rx="10" fill={t.bgCardSolid} stroke="#38bdf8" strokeWidth="1.5" strokeDasharray="4" />
          <text x="750" y="14" textAnchor="middle" fill="#38bdf8" fontSize="10" fontWeight="700" fontFamily="'Fira Code',monospace">PHASE 3: VERIFICATION</text>
          <rect x="630" y="45" width="90" height="28" rx="6" fill="#38bdf820" stroke="#38bdf8" strokeWidth="1" />
          <text x="675" y="63" textAnchor="middle" fill={t.text} fontSize="9">VLM Output</text>
          <rect x="740" y="45" width="90" height="28" rx="6" fill="#38bdf820" stroke="#38bdf8" strokeWidth="1" />
          <text x="785" y="63" textAnchor="middle" fill={t.text} fontSize="9">BiomedCLIP</text>
          <line x1="720" y1="59" x2="740" y2="59" stroke={t.textMuted} strokeWidth="1" />
          <rect x="680" y="90" width="100" height="28" rx="6" fill="#38bdf820" stroke="#38bdf8" strokeWidth="1" />
          <text x="730" y="108" textAnchor="middle" fill={t.text} fontSize="9">Cross-Check</text>
          <rect x="640" y="140" width="80" height="26" rx="5" fill="#22c55e15" stroke="#22c55e" strokeWidth="1" />
          <text x="680" y="157" textAnchor="middle" fill="#22c55e" fontSize="8">✓ Verified</text>
          <rect x="740" y="140" width="80" height="26" rx="5" fill="#ef444415" stroke="#ef4444" strokeWidth="1" />
          <text x="780" y="157" textAnchor="middle" fill="#ef4444" fontSize="8">⚠ Flagged</text>
          <rect x="700" y="178" width="80" height="20" rx="4" fill="#22c55e20" stroke="#22c55e" strokeWidth="1" />
          <text x="740" y="192" textAnchor="middle" fill="#22c55e" fontSize="8">Final Answer</text>

          {/* Arrowhead defs */}
          <defs>
            <marker id="arrowM" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
              <path d="M0,0 L8,3 L0,6" fill={t.textMuted} />
            </marker>
            <marker id="arrowG" markerWidth="6" markerHeight="5" refX="6" refY="2.5" orient="auto">
              <path d="M0,0 L6,2.5 L0,5" fill="#00c9a7" />
            </marker>
          </defs>
        </svg>
      </div>

      {/* Results Dashboard */}
      {metrics && (
        <section style={{ marginBottom: 32 }}>
          <h2 style={{ fontFamily: "'DM Serif Display',serif", fontSize: 22, color: t.textHeading, marginBottom: 16 }}>
            Results Dashboard
          </h2>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(340px,1fr))", gap: 16 }}>
            {/* Grouped bar chart */}
            <div style={cardStyle}>
              <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 12, color: t.textHeading }}>Performance Comparison</h3>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={["ECS", "GSS", "CHAIRs", "Refusal_Accuracy", "Entity_F1", "BERTScore"].map(k => ({
                  name: k.replace("_", " "),
                  "Zero-shot": metrics.zero_shot?.[k],
                  "ECF-DPO": metrics.ecf_dpo?.[k],
                  "+Verifier": metrics.ecf_dpo_with_verifier?.[k],
                }))}>
                  <CartesianGrid strokeDasharray="3 3" stroke={t.border} />
                  <XAxis dataKey="name" tick={{ fill: t.textMuted, fontSize: 10 }} angle={-25} textAnchor="end" height={50} />
                  <YAxis tick={{ fill: t.textMuted, fontSize: 10 }} domain={[0, 1]} />
                  <Tooltip contentStyle={{ background: t.bgCardSolid, border: `1px solid ${t.border}`, borderRadius: 8, fontSize: 12 }} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Bar dataKey="Zero-shot" fill="#6b7280" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="ECF-DPO" fill="#00c9a7" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="+Verifier" fill="#38bdf8" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Curriculum line chart */}
            <div style={cardStyle}>
              <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 12, color: t.textHeading }}>Curriculum Progression</h3>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={metrics.curriculum_stages ? [
                  { stage: "Stage 1 Easy", ...metrics.curriculum_stages.stage1_easy },
                  { stage: "Stage 2 Hard", ...metrics.curriculum_stages.stage2_hard },
                  { stage: "Stage 3 Mixed", ...metrics.curriculum_stages.stage3_final },
                ] : []}>
                  <CartesianGrid strokeDasharray="3 3" stroke={t.border} />
                  <XAxis dataKey="stage" tick={{ fill: t.textMuted, fontSize: 10 }} />
                  <YAxis tick={{ fill: t.textMuted, fontSize: 10 }} domain={[0, 0.8]} />
                  <Tooltip contentStyle={{ background: t.bgCardSolid, border: `1px solid ${t.border}`, borderRadius: 8, fontSize: 12 }} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Line type="monotone" dataKey="ECS" stroke="#00c9a7" strokeWidth={2} dot={{ r: 4 }} />
                  <Line type="monotone" dataKey="Refusal_Accuracy" stroke="#f97316" strokeWidth={2} dot={{ r: 4 }} name="Refusal Acc" />
                  <Line type="monotone" dataKey="Entity_F1" stroke="#38bdf8" strokeWidth={2} dot={{ r: 4 }} name="Entity F1" />
                </LineChart>
              </ResponsiveContainer>
            </div>


            {/* Verifier stats */}
            {metrics.verifier_stats && (
              <div style={cardStyle}>
                <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 16, color: t.textHeading }}>
                  <Icon name="shield" size={16} color={t.accent} /> Entity Verifier Performance
                </h3>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                  {[
                    { l: "Macro F1", v: metrics.verifier_stats.classifier_macro_f1 },
                    { l: "Precision", v: `${metrics.verifier_stats.verification_precision}%` },
                    { l: "Polyp F1", v: metrics.verifier_stats.polyp_f1 },
                    { l: "UC F1", v: metrics.verifier_stats.uc_f1 },
                    { l: "Oesophagitis F1", v: metrics.verifier_stats.oesophagitis_f1 },
                  ].map((s, i) => (
                    <div key={i} style={{ padding: "10px 12px", background: `${t.accent}08`, borderRadius: 8, border: `1px solid ${t.border}` }}>
                      <div style={{ fontSize: 11, color: t.textMuted }}>{s.l}</div>
                      <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 22, fontWeight: 700, color: t.accent }}>{s.v}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </section>
      )}

    </div>
  );
}
