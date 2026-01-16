import { pipeline } from "https://cdn.jsdelivr.net/npm/@xenova/transformers/dist/transformers.min.js";

// --------------------
// Config
// --------------------
const MODEL_ID = "Xenova/all-MiniLM-L6-v2"; // good quality, runs in browser
const WIN_SIM = 0.78;
const MIN_SIM = 0.05;

const CORE_R = 60;
const OUTER_R = 390;

// Daily target list 
const TARGETS = [
  // physical / natural
"meteor","comet","asteroid","nebula","cosmos","vacuum","plasma","quasar",
"monsoon","avalanche","earthquake","aftershock","fissure","rift","canyon",
"delta","estuary","lagoon","reef","abyss","trench","geyser","thermal",
"frost","hail","dew","drought","blight","canopy","understory","root",
"ember","ash","smoke","soot","flare","spark","wildfire","char",


  // places / structures
  "bastion","citadel","fortress","outpost","encampment","tower","spire",
"basement","attic","cellar","vault","archive","repository","chamber",
"gallery","hallway","alcove","atrium","courtyard","plaza","market",
"border","frontier","checkpoint","terminal","platform","station",
"monument","shrine","crypt","mausoleum","graveyard","lighthouse",

  // emotions / inner states
 "yearning","regret","envy","guilt","shame","pride","relief","panic",
"contentment","serenity","unease","restlessness","apathy","resentment",
"affection","devotion","adoration","tenderness","bitterness","jealousy",
"despair","emptiness","fulfillment","loneliness","belonging","alienation",
"gratitude","compassion","empathy","detachment","vulnerability",

  // abstract concepts
  "duality","paradox","infinity","finitude","continuity","discontinuity",
"causality","randomness","probability","certainty","ambiguity","clarity",
"origin","destination","transition","boundary","threshold","liminality",
"symmetry","asymmetry","equilibrium","instability","emergence","collapse",
"truth","illusion","appearance","essence","potential","actuality",

  // human / social
 "alliance","betrayal","loyalty","authority","rebellion","obedience",
"tradition","heritage","custom","taboo","identity","reputation",
"status","hierarchy","community","isolation","solidarity","division",
"cooperation","competition","negotiation","compromise","sacrifice",
"legacy","inheritance","mentorship","leadership","followership",

  // creative / intellectual
 "composition","structure","form","contrast","tone","texture","palette",
"motif","theme","variation","iteration","draft","revision","edit",
"critique","interpretation","expression","abstraction","minimalism",
"maximalism","innovation","tradition","influence","canon","experiment",
"play","practice","discipline","mastery","craftsmanship",

  // science / tech flavored
 "algorithm","protocol","architecture","interface","latency","bandwidth",
"signal","noise","entropy","compression","resolution","precision",
"approximation","iteration","optimization","convergence","divergence",
"variable","parameter","constraint","dataset","distribution","outlier",
"feedback","control","automation","simulation","modeling","prediction",

// Time / Process
"moment","instant","duration","interval","sequence","cycle","loop",
"phase","epoch","era","aftermath","prelude","aftermath",
"beginning","ending","delay","pause","acceleration","decay",
"growth","erosion","drift","accumulation","release",

// Sensory / Atmospheric
"silence","whisper","hum","static","reverberation","glow","flicker",
"shadow","glare","blur","haze","fog","scent","fragrance","stench",
"warmth","chill","pressure","weight","lightness","roughness","smoothness",

// Weird
"ghost","trace","scar","imprint","residue","fragment","shard","relic",
"echo","mirror","reflection","veil","mask","pulse","rift",
"thread","knot","tangle","web","loop","spiral","axis","center","edge",


];


// --------------------
// DOM
// --------------------
const canvas = document.getElementById("c");
const ctx = canvas.getContext("2d");
const input = document.getElementById("guess");
const logEl = document.getElementById("log");
const puzzleEl = document.getElementById("puzzleId");
const guessCountEl = document.getElementById("guessCount");

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

function similarityToRadius(sim) {
  sim = clamp(sim, -1, 1);
  if (sim < MIN_SIM) return OUTER_R;

  const t = clamp((sim - MIN_SIM) / (1 - MIN_SIM), 0, 1);
  const eased = 1 - Math.pow(t, 1.8);
  return CORE_R + eased * (OUTER_R - CORE_R);
}

function motionKind(sim) {
  if (sim >= 0.32) return "tight";
  if (sim >= 0.27) return "pulled";
  if (sim >= 0.20) return "drift";
  if (sim >= 0.10) return "wobble";
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

function pickDailyTarget() {
  const iso = new Date().toISOString().slice(0, 10); // UTC day
  const seed = hash32(iso);
  const idx = seed % TARGETS.length;
  PUZZLE_ID = `${iso}-${seed}`;
  puzzleEl.textContent = PUZZLE_ID;
  targetWord = TARGETS[idx];
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


// simple angle from embedding chunks (cheap projection)
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
    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    ctx.lineWidth = (r === CORE_R) ? 2 : 1;
    ctx.stroke();
  }

  ctx.beginPath();
  ctx.moveTo(-OUTER_R, 0);
  ctx.lineTo(OUTER_R, 0);
  ctx.moveTo(0, -OUTER_R);
  ctx.lineTo(0, OUTER_R);
  ctx.strokeStyle = "rgba(255,255,255,0.04)";
  ctx.stroke();

  const extra = solved ? (18 * collapseT) : 0;
  ctx.beginPath();
  ctx.arc(0,0, (CORE_R-6) + extra, 0, Math.PI*2);
  ctx.fillStyle = solved ? "rgba(210,210,255,0.18)" : "rgba(170,170,255,0.10)";
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
  // returns a 384-d embedding vector (array of floats)
  const out = await embedder(text, { pooling: "mean", normalize: true });
  // out.data is a Float32Array
  return Array.from(out.data);
}

async function init() {
  addLog("Loading model (first time may take a bit)...", "");
  embedder = await pipeline("feature-extraction", MODEL_ID);
  pickDailyTarget();
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

  await applyGuess(word, false);

});

init();
