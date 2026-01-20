import crypto from "node:crypto";

// ---- keep this TARGETS list identical to orbit.js ----
const TARGETS = [
  // ...paste the exact same TARGETS array from orbit.js here...
];

// format: YYYY-MM-DD (local date)
function todayKey(d = new Date()) {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

function pickDailyTarget(dateKey) {
  const hashHex = crypto.createHash("sha256").update(dateKey).digest("hex");
  // take first 8 hex chars -> 32-bit number
  const n = parseInt(hashHex.slice(0, 8), 16);
  return TARGETS[n % TARGETS.length];
}

const key = todayKey();
const answer = pickDailyTarget(key);

console.log("[SO DEBUG]");
console.log("dateKey:", key);
console.log("answer :", answer);
