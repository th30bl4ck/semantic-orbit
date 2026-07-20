import { pipeline } from "https://cdn.jsdelivr.net/npm/@xenova/transformers/dist/transformers.min.js";

// --------------------
// Config
// --------------------
const MODEL_ID = "Xenova/all-MiniLM-L6-v2"; 
const WIN_SIM = 0.90;
const MIN_SIM = 0.05;

const CORE_R = 60;
const OUTER_R = 390;

let won = false;
let winSeed = null;


const TARGETS = [

  // ─── Physical / Natural ─────────────────────────────
  "meteor","comet","asteroid","nebula","supernova","pulsar","quasar","vacuum",
  "plasma","gravity","radiation","orbit","eclipse","flare","fusion","collapse",
  "monsoon","cyclone","avalanche","landslide","earthquake","tremor","fissure",
  "canyon","gorge","plateau","delta","estuary","lagoon","reef","abyss",
  "trench","geyser","thermal","frost","hail","dew","drought","blight","canopy",
  "understory","root","ember","ash","smoke","soot","spark","wildfire","char",
  "erosion","sediment","mineral","current","tide","horizon","atmosphere",
  "pressure","turbulence","updraft","fallout","resonance",

  // ─── Built / Places / Structures ────────────────────
  "bastion","citadel","fortress","outpost","encampment","tower","spire",
  "basement","attic","cellar","vault","archive","repository","chamber",
  "gallery","corridor","alcove","atrium","courtyard","plaza","market",
  "border","frontier","checkpoint","terminal","platform","station",
  "monument","shrine","crypt","mausoleum","graveyard","lighthouse",
  "observatory","sanctuary","barracks","depot","armory","warehouse",
  "workshop","studio","library","conservatory","greenhouse","silo",
  "aqueduct","viaduct","bridge","passage","threshold","enclosure",

  // ─── Emotional / Psychological ──────────────────────
  "yearning","regret","envy","guilt","shame","pride","relief","panic",
  "contentment","serenity","unease","restlessness","apathy","resentment",
  "affection","devotion","adoration","tenderness","bitterness","jealousy",
  "despair","emptiness","fulfillment","loneliness","belonging","alienation",
  "gratitude","compassion","empathy","detachment","vulnerability",
  "anticipation","dread","melancholy","nostalgia","wonder","awe",
  "frustration","irritation","exhaustion","calm","resolve",
  "acceptance","denial","fixation","doubt","reassurance",

  // ─── Abstract Concepts ──────────────────────────────
  "duality","paradox","infinity","finitude","continuity","discontinuity",
  "causality","randomness","probability","certainty","ambiguity","clarity",
  "origin","destination","transition","boundary","liminality",
  "symmetry","asymmetry","equilibrium","instability","emergence",
  "truth","illusion","appearance","essence","potential","actuality",
  "identity","meaning","absence","presence","coherence","contradiction",
  "structure","chaos","order","pattern","anomaly","context","perspective",
  "intention","consequence","interpretation","abstraction",

  // ─── Human / Social ─────────────────────────────────
  "alliance","betrayal","loyalty","authority","rebellion","obedience",
  "heritage","custom","taboo","reputation","status",
  "hierarchy","community","isolation","solidarity","division",
  "cooperation","competition","negotiation","compromise","sacrifice",
  "legacy","inheritance","mentorship","leadership","followership",
  "conformity","dissent","influence","persuasion","manipulation",
  "trust","suspicion","accountability","obligation","consensus",
  "polarization","affiliation","exclusion",

  // ─── Creative / Intellectual ────────────────────────
  "composition","form","contrast","tone","texture","palette",
  "motif","theme","variation","draft","revision","edit","critique",
  "expression","minimalism","maximalism","innovation",
  "canon","experiment","play","practice","discipline",
  "mastery","craftsmanship","intuition","exploration","synthesis",
  "distortion","emphasis","rhythm","balance","harmony",

  // ─── Science / Technology ───────────────────────────
  "algorithm","protocol","architecture","interface","latency","bandwidth",
  "signal","noise","entropy","compression","resolution","precision",
  "approximation","optimization","convergence","divergence",
  "variable","parameter","constraint","dataset","distribution",
  "outlier","feedback","control","automation","simulation",
  "modeling","prediction","calibration","inference","scalability",
  "redundancy","robustness","failure",

  // ─── Time / Process / Change ────────────────────────
  "moment","instant","duration","interval","sequence","cycle","phase",
  "epoch","era","prelude","aftermath","beginning","ending","delay",
  "pause","acceleration","decay","growth","drift","accumulation",
  "release","progression","regression","repetition","disruption",
  "stagnation","momentum","inertia","culmination",
  "interruption","renewal","transformation",

  // ─── Sensory / Atmospheric ──────────────────────────
  "silence","whisper","hum","static","reverberation","glow","flicker",
  "shadow","glare","blur","haze","fog","scent","fragrance","warmth",
  "chill","weight","lightness","roughness","smoothness",
  "vibration","echo","murmur","stillness","shimmer",
  "dullness","sharpness","density","openness",

  // ─── Spatial / Relational ───────────────────────────
  "center","edge","axis","alignment","offset","distance","proximity",
  "overlap","separation","exposure","depth","surface",
  "layer","margin","void","frame","crossing","approach",
  "retreat","orientation","scale","position",

  // ─── Existential / Philosophical ────────────────────
  "mortality","impermanence","selfhood","agency",
  "freewill","determinism","responsibility","awareness",
  "consciousness","perception","subjectivity","objectivity",
  "purpose","belief","faith","skepticism",
  "absurdity","transcendence",

  // ─── Weird / Liminal / Orbit-Core ───────────────────
  "ghost","trace","scar","imprint","residue","fragment","shard",
  "relic","mirror","reflection","veil","mask","thread","knot",
  "tangle","web","spiral","fracture","afterimage",
  "recursion","hollow","inversion","displacement",
  "riddle","cipher","artifact"

];


// --------------------
// Run signature 
// --------------------


function fnv1a32(str) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    
    h = (h + ((h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24))) >>> 0;
  }
  return h >>> 0;
}


function mulberry32(seed) {
  let a = seed >>> 0;
  return function rand() {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}


function seedFromRun(targetWord, guesses) {
  
  const parts = guesses.map(g => {
    const w = (g.word || "").toLowerCase().trim();
    const s = (g.sim ?? 0);
    const r = (g.r ?? 0);
    return `${w}:${s.toFixed(4)}:${Math.round(r)}`;
  });

  const signature = `target=${(targetWord || "").toLowerCase()}|` + parts.join("|");
  return fnv1a32(signature);
}

// --------------------
// Generative core art
// --------------------
function hsl(h, s, l, a = 1) {
  return `hsla(${h}, ${s}%, ${l}%, ${a})`;
}

function drawCoreArt(ctx, cx, cy, coreR, seed) {
  const rand = mulberry32(seed);

  ctx.save();
  // Clip to the core circle so art stays inside
  ctx.beginPath();
  ctx.arc(cx, cy, coreR, 0, Math.PI * 2);
  ctx.clip();

  // Background wash (soft blobs)
  ctx.globalCompositeOperation = "source-over";
  ctx.clearRect(cx - coreR, cy - coreR, coreR * 2, coreR * 2);

  const baseHue = Math.floor(rand() * 360);
  const blobCount = 10 + Math.floor(rand() * 14);

  for (let i = 0; i < blobCount; i++) {
    const hue = (baseHue + Math.floor(rand() * 140) - 70 + 360) % 360;
    const radius = coreR * (0.25 + rand() * 0.9);
    const x = cx + (rand() * 2 - 1) * coreR;
    const y = cy + (rand() * 2 - 1) * coreR;

    const grad = ctx.createRadialGradient(x, y, 0, x, y, radius);
    grad.addColorStop(0, hsl(hue, 85, 60, 0.55));
    grad.addColorStop(1, hsl(hue, 85, 20, 0.0));
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
  }

 
  ctx.globalCompositeOperation = "screen";
  const ringCount = 6 + Math.floor(rand() * 10);
  for (let i = 0; i < ringCount; i++) {
    const hue = (baseHue + Math.floor(rand() * 220)) % 360;
    const r = coreR * (0.15 + rand() * 0.95);
    const lw = 1 + rand() * 4;
    const start = rand() * Math.PI * 2;
    const end = start + (0.3 + rand() * 1.9) * Math.PI;

    ctx.strokeStyle = hsl(hue, 90, 65, 0.35 + rand() * 0.35);
    ctx.lineWidth = lw;
    ctx.beginPath();
    ctx.arc(cx, cy, r, start, end);
    ctx.stroke();
  }

  // Sprinkle shapes: dots, triangles, little “comets”
  ctx.globalCompositeOperation = "lighter";
  const shapeCount = 30 + Math.floor(rand() * 60);

  for (let i = 0; i < shapeCount; i++) {
    const hue = (baseHue + Math.floor(rand() * 180) - 90 + 360) % 360;
    const x = cx + (rand() * 2 - 1) * coreR;
    const y = cy + (rand() * 2 - 1) * coreR;

    const t = rand();
    if (t < 0.55) {
      // dot
      const r = 0.8 + rand() * 3.5;
      ctx.fillStyle = hsl(hue, 90, 70, 0.25 + rand() * 0.5);
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
    } else if (t < 0.80) {
      // triangle
      const size = 3 + rand() * 10;
      const ang = rand() * Math.PI * 2;
      ctx.fillStyle = hsl(hue, 95, 65, 0.18 + rand() * 0.35);
      ctx.beginPath();
      for (let k = 0; k < 3; k++) {
        const a = ang + k * (Math.PI * 2 / 3);
        const px = x + Math.cos(a) * size;
        const py = y + Math.sin(a) * size;
        if (k === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.closePath();
      ctx.fill();
    } else {
      // comet streak
      const len = 8 + rand() * 30;
      const ang = rand() * Math.PI * 2;
      const x2 = x + Math.cos(ang) * len;
      const y2 = y + Math.sin(ang) * len;

      ctx.strokeStyle = hsl(hue, 95, 70, 0.12 + rand() * 0.25);
      ctx.lineWidth = 1 + rand() * 2.5;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }
  }

  ctx.restore();


  ctx.save();
  ctx.globalCompositeOperation = "source-over";
  ctx.strokeStyle = "rgba(255,255,255,0.12)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(cx, cy, coreR - 1, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

if (won) {
  const seed = seedFromRun(targetWord, guesses);
  drawCoreArt(ctx, centerX, centerY, CORE_R, seed);
}


// --------------------
// DOM
// --------------------
const canvas = document.getElementById("c");
const ctx = canvas.getContext("2d");
const input = document.getElementById("guess");
const logEl = document.getElementById("log");
const puzzleEl = document.getElementById("puzzleId");
const guessCountEl = document.getElementById("guessCount");
const resetGuessesBtn = document.getElementById("resetGuesses");

const WORLD = { w: canvas.width, h: canvas.height };
const CENTER = { x: WORLD.w / 2, y: WORLD.h / 2 };

// --------------------
// State
// --------------------
let solved = false;
let collapseT = 0;
let PUZZLE_ID = null;

const nodes = []; // {word, x,y, tx,ty, vx,vy, motion, win, born, seed, baseAngle, radius, orbitSpeed}
let embedder = null;
let targetWord = null;
let targetVec = null;
let progress = { puzzleId: null, guesses: [] };

const STORAGE_KEY = "semantic-orbit-progress";


let streak = null;


const STREAK_KEY = "semantic-orbit-streak-v1";

function londonDayId(d = new Date()) {
 
  const fmt = new Intl.DateTimeFormat("en-CA", {
    timeZone: "Europe/London",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  });
  return fmt.format(d);
}

function loadStreak() {
  try {
    return JSON.parse(localStorage.getItem(STREAK_KEY) || "null") || {
      currentStreak: 0,
      bestStreak: 0,
      lastWinDayId: null,   // "YYYY-MM-DD"
      lastSeenDayId: null,  // "YYYY-MM-DD"
      wins: 0,
    };
  } catch {
    return {
      currentStreak: 0,
      bestStreak: 0,
      lastWinDayId: null,
      lastSeenDayId: null,
      wins: 0,
    };
  }
}

function saveStreak(s) {
  localStorage.setItem(STREAK_KEY, JSON.stringify(s));
}


function onGameOpenToday() {
  const s = loadStreak();
  const today = londonDayId();

  if (s.lastSeenDayId && s.lastSeenDayId !== today) {
    const y = new Date();
    y.setDate(y.getDate() - 1);
    const yesterday = londonDayId(y);

    
    if (s.lastWinDayId !== yesterday) {
      s.currentStreak = 0;
    }
  }

  s.lastSeenDayId = today;
  saveStreak(s);
  return s;
}


function onWinToday() {
  const s = loadStreak();
  const today = londonDayId();

 
  if (s.lastWinDayId === today) return s;

  const y = new Date();
  y.setDate(y.getDate() - 1);
  const yesterday = londonDayId(y);

  if (s.lastWinDayId === yesterday) {
    s.currentStreak += 1;
  } else {
    s.currentStreak = 1;
  }

  s.bestStreak = Math.max(s.bestStreak, s.currentStreak);
  s.lastWinDayId = today;
  s.wins += 1;

  saveStreak(s);
  return s;
}

// --------------------
// Helpers
// --------------------
function addLog(text, cls="") {
  const d = document.createElement("div");
  d.textContent = text;
  if (cls) d.className = cls;
  logEl.prepend(d);
}

function updateGuessCount() {
  guessCountEl.textContent = `Guesses: ${progress.guesses.length}`;
}

function norm(v) {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i]*v[i];
  const n = Math.sqrt(s) || 1e-9;
  return v.map(x => x / n);
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i]*b[i];
  return s;
}

function clamp(x, lo, hi) {
  return Math.max(lo, Math.min(hi, x));
}

function rand(min, max) {
  return Math.random() * (max - min) + min;
}

function similarityToRadius(sim) {
  sim = clamp(sim, -1, 1);
  if (sim < MIN_SIM) return OUTER_R;

    switch (motionKind(sim)) {
  case "tight":   return rand(70, 90);     // closest ring
  case "pulled":  return rand(100, 160);
  case "drift":   return rand(180, 240);
  case "wobble":  return rand(260, 320);
  case "pushed":  return rand(360, 390);   // outer ring
  default:        return rand(360, 390);
  }
}

function motionKind(sim) {
  if (sim >= 0.60) return "tight";
  if (sim >= 0.45) return "pulled";
  if (sim >= 0.35) return "drift";
  if (sim >= 0.25) return "wobble";
  return "pushed";
}

// stable-ish hash for daily seed
function hash32(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function londonYesterdayDayId() {
  const d = new Date();
  d.setDate(d.getDate() - 1);
  return londonDayId(d);
}

function pickTargetForDayId(dayId) {
  const seed = hash32(dayId);
  const idx = seed % TARGETS.length;
  return {
    dayId,
    seed,
    idx,
    word: TARGETS[idx],
    puzzleId: `${dayId}-${seed}`,
  };
}

function showYesterdayWord() {
  const y = pickTargetForDayId(londonYesterdayDayId());
  const el = document.getElementById("yWord");
  if (el) el.textContent = y.word;
}

function pickDailyTarget() {
  const today = pickTargetForDayId(londonDayId()); // "YYYY-MM-DD" London
  PUZZLE_ID = today.puzzleId;
  puzzleEl.textContent = PUZZLE_ID;
  targetWord = today.word;

  window.__SO_DEBUG_TARGET = String(today.word);
}

function loadProgress() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { puzzleId: PUZZLE_ID, guesses: [] };
    const data = JSON.parse(raw);
    if (!data || data.puzzleId !== PUZZLE_ID || !Array.isArray(data.guesses)) {
      return { puzzleId: PUZZLE_ID, guesses: [] };
    }
    return { puzzleId: data.puzzleId, guesses: [...data.guesses] };
  } catch (err) {
    return { puzzleId: PUZZLE_ID, guesses: [] };
  }
}

function saveProgress() {
  const payload = {
    puzzleId: progress.puzzleId,
    guesses: progress.guesses,
  };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
}

function resetGuesses() {
  nodes.length = 0;
  progress.guesses = [];
  solved = false;
  collapseT = 0;
  logEl.innerHTML = "";
  saveProgress();
  updateGuessCount();
  addLog("Guesses cleared.", "");
}


function embeddingToAngle(vec) {
  // split into 2 sums to get a stable direction
  let x = 0, y = 0;
  for (let i = 0; i < vec.length; i++) {
    if (i % 2 === 0) x += vec[i];
    else y += vec[i];
  }
  return Math.atan2(y, x);
}

function spawnNode(word, targetX, targetY, motion, win, restoring=false) {
  const spawnAngle = Math.random() * Math.PI * 2;
  const spawnR = restoring ? (OUTER_R + 10) : (OUTER_R + 40);

  const sx = spawnR * Math.cos(spawnAngle);
  const sy = spawnR * Math.sin(spawnAngle);
  const baseAngle = Math.atan2(targetY, targetX);
  const radius = Math.hypot(targetX, targetY);
  const orbitSpeed = (Math.random() < 0.5 ? -1 : 1) * motionParams(motion).orbitSpeed;

  nodes.push({
    word,
    x: sx, y: sy,
    tx: targetX, ty: targetY,
    vx: 0, vy: 0,
    motion,
    win,
    born: performance.now(),
    seed: Math.random() * 10,
    baseAngle,
    radius,
    orbitSpeed
  });
}

function motionParams(motion) {
  switch(motion) {
    case "tight":  return { spring: 0.12, damp: 0.85, jitter: 0.06, orbitSpeed: 0.10 };
    case "pulled": return { spring: 0.10, damp: 0.86, jitter: 0.08, orbitSpeed: 0.08 };
    case "drift":  return { spring: 0.07, damp: 0.88, jitter: 0.10, orbitSpeed: 0.06 };
    case "wobble": return { spring: 0.05, damp: 0.90, jitter: 0.12, orbitSpeed: 0.05 };
    case "pushed": return { spring: 0.04, damp: 0.92, jitter: 0.14, orbitSpeed: 0.04 };
    default:       return { spring: 0.06, damp: 0.88, jitter: 0.10, orbitSpeed: 0.06 };
  }
}

function drawRings() {
  ctx.save();
  ctx.translate(CENTER.x, CENTER.y);

  for (let r of [CORE_R, 140, 220, 300, OUTER_R]) {
    ctx.beginPath();
    ctx.arc(0,0,r,0,Math.PI*2);
    ctx.strokeStyle = "rgba(255,255,255,0.16)";
    ctx.lineWidth = (r === CORE_R) ? 2 : 1;
    ctx.stroke();
  }

  ctx.beginPath();
  ctx.moveTo(-OUTER_R, 0);
  ctx.lineTo(OUTER_R, 0);
  ctx.moveTo(0, -OUTER_R);
  ctx.lineTo(0, OUTER_R);
  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.stroke();

  const extra = solved ? (18 * collapseT) : 0;
  ctx.beginPath();
  ctx.arc(0,0, (CORE_R-6) + extra, 0, Math.PI*2);
  ctx.fillStyle = solved ? "rgba(210,210,255,0.18)" : "rgba(170,170,255,0.22)";
  ctx.fill();

  ctx.restore();
}

function worldToScreen(wx, wy) {
  return { x: CENTER.x + wx, y: CENTER.y + wy };
}

function update(dt) {
  if (solved && collapseT < 1) collapseT = Math.min(1, collapseT + 0.02 * dt);

  for (const n of nodes) {
    const t = (performance.now() - n.born) * 0.001;
    const orbitAngle = n.baseAngle + n.orbitSpeed * t;
    n.tx = n.radius * Math.cos(orbitAngle);
    n.ty = n.radius * Math.sin(orbitAngle);

    const {spring, damp, jitter} = motionParams(n.motion);
    const ax = (n.tx - n.x) * spring;
    const ay = (n.ty - n.y) * spring;

    const jx = Math.sin(t*2.1 + n.seed) * jitter;
    const jy = Math.cos(t*1.7 + n.seed*0.7) * jitter;

    n.vx = (n.vx + ax + jx) * damp;
    n.vy = (n.vy + ay + jy) * damp;

    n.x += n.vx * dt;
    n.y += n.vy * dt;

    if (solved) {
      n.radius *= (1 - 0.02 * collapseT);
    }
  }
}

function draw() {
  ctx.clearRect(0,0,canvas.width,canvas.height);
  drawRings();

  for (const n of nodes) {
    const s = worldToScreen(n.x, n.y);
    const age = Math.min(1, (performance.now() - n.born)/600);

    ctx.beginPath();
    ctx.arc(s.x, s.y, 10 + 6*(1-age), 0, Math.PI*2);
    ctx.fillStyle = "rgba(200,200,255,0.10)";
    ctx.fill();

    ctx.beginPath();
    ctx.arc(s.x, s.y, 5, 0, Math.PI*2);
    ctx.fillStyle = n.win ? "rgba(255,255,255,0.95)" : "rgba(255,255,255,0.75)";
    ctx.fill();

    ctx.font = "13px system-ui";
    ctx.fillStyle = "rgba(255,255,255,0.85)";
    ctx.fillText(n.word, s.x + 10, s.y - 10);
  }

  if (solved) {
    ctx.save();
    ctx.fillStyle = "rgba(255,255,255,0.92)";
    ctx.font = "bold 28px system-ui";
    ctx.fillText("CORE FOUND", 18, 44);
    ctx.font = "14px system-ui";
    ctx.fillStyle = "rgba(255,255,255,0.75)";
    ctx.fillText("Come back tomorrow for a new orbit.", 18, 68);
    ctx.restore();
  }
}

let last = performance.now();
function loop() {
  const now = performance.now();
  const dt = Math.min(2.0, (now - last) / 16.67);
  last = now;
  update(dt);
  draw();
  requestAnimationFrame(loop);
}
loop();

// --------------------
// Embeddings
// --------------------
async function embed(text) {
  
  const out = await embedder(text, { pooling: "mean", normalize: true });

  return Array.from(out.data);
}

async function init() {
  addLog("Loading model (first time may take a bit)...", "");
  embedder = await pipeline("feature-extraction", MODEL_ID);

  pickDailyTarget();
  showYesterdayWord();
  
  streak = onGameOpenToday();
  addLog(`Streak: ${streak.currentStreak} (best ${streak.bestStreak})`, "");

  targetVec = await embed(targetWord);

  progress = loadProgress();
  updateGuessCount();

  if (progress.guesses.length > 0) {
    addLog(`Restoring ${progress.guesses.length} guess${progress.guesses.length === 1 ? "" : "es"}...`, "");
    for (const word of progress.guesses) {
      await applyGuess(word, true);
    }
  }
  addLog("Ready.", "");
}

async function applyGuess(word, restoring = false) {
  const v = await embed(word);
  const sim = dot(v, targetVec); // already normalized
  const r = similarityToRadius(sim);
  const ang = embeddingToAngle(v);

  const x = r * Math.cos(ang);
  const y = r * Math.sin(ang);

  const win = sim >= WIN_SIM || r <= CORE_R + 4;
  const motion = motionKind(sim);

  spawnNode(word, x, y, motion, win, restoring);
  addLog(`${word} → ${motion}${win ? " (core!)" : ""}`, win ? "solved" : "");

  if (win) {
    solved = true;

    
    streak = onWinToday();
    addLog(`Streak is now ${streak.currentStreak} (best ${streak.bestStreak})`, "solved");

    addLog("You entered the core.", "solved");
  }

  if (!restoring) {
    progress.guesses.push(word);
    saveProgress();
    updateGuessCount();
  }
}

input.addEventListener("keydown", async (e) => {
  if (e.key !== "Enter") return;
  if (!embedder) return;
  if (solved) return;

  const word = input.value.trim().toLowerCase();
  if (!word) return;
  input.value = "";

  if (progress.guesses.includes(word)) {
    addLog("already guessed", "");
    return;
  }

  await applyGuess(word, false);
});

resetGuessesBtn.addEventListener("click", () => {
  if (!embedder) return;
  resetGuesses();
});

init();
