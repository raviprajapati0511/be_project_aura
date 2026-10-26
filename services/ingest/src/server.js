// AURA ingestion service - Express + Socket.io + MySQL.
//
// Accepts telemetry samples from the simulator over HTTP POST /ingest,
// inserts into the MySQL telemetry table, and broadcasts each sample
// to connected dashboard clients over Socket.io.

import express from "express";
import fs from "node:fs";
import { createServer } from "node:http";
import path from "node:path";
import { Server } from "socket.io";

import mysql from "mysql2/promise";

function loadRootEnv() {
  const candidates = [
    path.resolve(process.cwd(), ".env"),
    path.resolve(process.cwd(), "..", ".env"),
    path.resolve(process.cwd(), "..", "..", ".env"),
  ];
  const envPath = candidates.find((candidate) => fs.existsSync(candidate));
  if (!envPath) return;

  for (const line of fs.readFileSync(envPath, "utf8").split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#") || !trimmed.includes("=")) continue;
    const [key, ...rest] = trimmed.split("=");
    if (!process.env[key]) {
      process.env[key] = rest.join("=").trim().replace(/^['"]|['"]$/g, "");
    }
  }
}

loadRootEnv();

const PORT = Number(process.env.INGEST_PORT ?? 4000);
const databaseUrl = new URL(
  process.env.DATABASE_URL ?? "mysql://root@localhost:3306/fitness_db",
);

const pool = mysql.createPool({
  host: databaseUrl.hostname,
  port: Number(databaseUrl.port || 3306),
  user: decodeURIComponent(databaseUrl.username),
  password: decodeURIComponent(databaseUrl.password),
  database: databaseUrl.pathname.slice(1),
  connectTimeout: 2000,
  connectionLimit: 5,
});

const app = express();
app.use(express.json({ limit: "64kb" }));

const httpServer = createServer(app);
const io = new Server(httpServer, {
  cors: { origin: "*", methods: ["GET", "POST"] },
});

const securityCounters = { rejected_payloads: 0, rate_limited: 0, malformed_json: 0, last_rejected_at: null };
const ingestWindows = new Map();

app.use("/ingest", (req, res, next) => {
  const now = Date.now();
  const client = req.socket.remoteAddress || "unknown";
  let window = ingestWindows.get(client);
  if (!window || now - window.started_at >= 60_000) {
    window = { started_at: now, requests: 0 };
    ingestWindows.set(client, window);
  }
  window.requests += 1;
  if (ingestWindows.size > 10000) {
    for (const [key, value] of ingestWindows) if (now - value.started_at >= 60_000) ingestWindows.delete(key);
  }
  if (window.requests > 120) {
    securityCounters.rate_limited += 1;
    res.status(429).json({ error: "rate_limit_exceeded" });
    return;
  }
  next();
});

app.get("/health", (_req, res) => {
  res.json({ status: "ok", ts: new Date().toISOString() });
});

app.get("/healthz", async (_req, res) => {
  try {
    await pool.query("SELECT 1");
    res.json({ status: "ok", db: true });
  } catch {
    res.status(503).json({ status: "degraded", db: false });
  }
});

app.get("/security/summary", (_req, res) => {
  res.json({ ...securityCounters, scope: "local ingestion endpoint and schema validation only" });
});

function validateTelemetry(body) {
  if (!body || typeof body !== "object" || Array.isArray(body)) return null;
  if (typeof body.vehicle_id !== "string" || !body.vehicle_id.trim()) return null;
  const need = [
    "time",
    "vehicle_id",
    "pack_voltage",
    "pack_current",
    "soc_reported",
    "cell_temp_max",
    "speed_kmh",
    "odometer_km",
    "latitude",
    "longitude",
  ];
  for (const key of need) {
    if (body[key] === undefined || body[key] === null) return null;
  }
  const numericFields = [
    "pack_voltage", "pack_current", "soc_reported", "cell_temp_max",
    "speed_kmh", "odometer_km", "latitude", "longitude",
    "battery_soh", "motor_temp", "motor_vibration", "power_consumption",
    "component_health", "failure_probability", "rul_score", "acceleration_mps2",
    "can_anomaly_score",
  ];
  if (numericFields.some((key) => typeof body[key] === "boolean" || (typeof body[key] === "string" && !body[key].trim()))) return null;

  const time = new Date(body.time);
  if (Number.isNaN(time.getTime())) return null;

  const row = {
    time,
    vehicle_id: String(body.vehicle_id).trim(),
    pack_voltage: Number(body.pack_voltage),
    pack_current: Number(body.pack_current),
    soc_reported: Number(body.soc_reported),
    cell_temp_max: Number(body.cell_temp_max),
    speed_kmh: Number(body.speed_kmh),
    odometer_km: Number(body.odometer_km),
    latitude: Number(body.latitude),
    longitude: Number(body.longitude),
    battery_soh: body.battery_soh == null ? null : Number(body.battery_soh),
    motor_temp: body.motor_temp == null ? null : Number(body.motor_temp),
    motor_vibration: body.motor_vibration == null ? null : Number(body.motor_vibration),
    power_consumption: body.power_consumption == null ? null : Number(body.power_consumption),
    component_health: body.component_health == null ? null : Number(body.component_health),
    failure_probability: body.failure_probability == null ? null : Number(body.failure_probability),
    rul_score: body.rul_score == null ? null : Number(body.rul_score),
    acceleration_mps2: body.acceleration_mps2 == null ? null : Number(body.acceleration_mps2),
    can_anomaly_score: body.can_anomaly_score == null ? null : Number(body.can_anomaly_score),
    maintenance_type: body.maintenance_type == null ? null : String(body.maintenance_type),
  };

  const numericKeys = [
    "pack_voltage",
    "pack_current",
    "soc_reported",
    "cell_temp_max",
    "speed_kmh",
    "odometer_km",
    "latitude",
    "longitude",
    "battery_soh",
    "motor_temp",
    "motor_vibration",
    "power_consumption",
    "component_health",
    "failure_probability",
    "rul_score",
    "acceleration_mps2",
    "can_anomaly_score",
  ];
  if (!row.vehicle_id || row.vehicle_id.length > 64) return null;
  if (numericKeys.some((key) => row[key] != null && !Number.isFinite(row[key]))) return null;
  if (row.pack_voltage < 0 || row.pack_voltage > 600) return null;
  if (row.pack_current < -400 || row.pack_current > 400) return null;
  if (row.soc_reported < 0 || row.soc_reported > 100) return null;
  if (row.cell_temp_max < -50 || row.cell_temp_max > 120) return null;
  if (row.speed_kmh < 0 || row.speed_kmh > 400 || row.odometer_km < 0) return null;
  if (row.latitude < -90 || row.latitude > 90 || row.longitude < -180 || row.longitude > 180) return null;
  if ([row.battery_soh, row.component_health, row.failure_probability].some((v) => v != null && (v < 0 || v > 1))) return null;
  if (row.can_anomaly_score != null && (row.can_anomaly_score < 0 || row.can_anomaly_score > 1)) return null;
  if ([row.motor_temp, row.motor_vibration, row.power_consumption, row.rul_score].some((v) => v != null && v < 0)) return null;
  if (row.acceleration_mps2 != null && Math.abs(row.acceleration_mps2) > 20) return null;
  if (row.maintenance_type != null && row.maintenance_type.length > 32) return null;
  return row;
}

const INSERT_SQL = `
  INSERT INTO vehicle_telemetry
    (time, vehicle_id, pack_voltage, pack_current, soc_reported,
     cell_temp_max, speed_kmh, odometer_km, latitude, longitude,
     battery_soh, motor_temp, motor_vibration, power_consumption,
     component_health, failure_probability, rul_score, maintenance_type, acceleration_mps2, can_anomaly_score)
  VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
`;

function socketPayload(row) {
  return {
    time: row.time.toISOString(),
    vehicle_id: row.vehicle_id,
    pack_voltage: row.pack_voltage,
    pack_current: row.pack_current,
    soc_reported: row.soc_reported,
    cell_temp_max: row.cell_temp_max,
    speed_kmh: row.speed_kmh,
    odometer_km: row.odometer_km,
    latitude: row.latitude,
    longitude: row.longitude,
    battery_soh: row.battery_soh,
    motor_temp: row.motor_temp,
    motor_vibration: row.motor_vibration,
    power_consumption: row.power_consumption,
    component_health: row.component_health,
    failure_probability: row.failure_probability,
    rul_score: row.rul_score,
    maintenance_type: row.maintenance_type,
    acceleration_mps2: row.acceleration_mps2,
    can_anomaly_score: row.can_anomaly_score,
  };
}

app.post("/ingest", async (req, res) => {
  const row = validateTelemetry(req.body);
  if (!row) {
    securityCounters.rejected_payloads += 1;
    securityCounters.last_rejected_at = new Date().toISOString();
    res.status(400).json({ error: "invalid telemetry payload" });
    return;
  }

  try {
    await pool.execute(INSERT_SQL, [
      row.time,
      row.vehicle_id,
      row.pack_voltage,
      row.pack_current,
      row.soc_reported,
      row.cell_temp_max,
      row.speed_kmh,
      row.odometer_km,
      row.latitude,
      row.longitude,
      row.battery_soh,
      row.motor_temp,
      row.motor_vibration,
      row.power_consumption,
      row.component_health,
      row.failure_probability,
      row.rul_score,
      row.maintenance_type,
      row.acceleration_mps2,
      row.can_anomaly_score,
    ]);
    io.emit("telemetry:new", socketPayload(row));
    res.status(201).json({ ok: true });
  } catch (e) {
    console.error("[ingest] insert failed:", e.message);
    res.status(503).json({ error: "db_unavailable" });
  }
});

app.use((error, _req, res, _next) => {
  if (error?.type === "entity.too.large") {
    securityCounters.rejected_payloads += 1;
    securityCounters.last_rejected_at = new Date().toISOString();
    res.status(413).json({ error: "payload_too_large" });
    return;
  }
  if (error instanceof SyntaxError && error.status === 400) {
    securityCounters.malformed_json += 1;
    securityCounters.last_rejected_at = new Date().toISOString();
    res.status(400).json({ error: "invalid_json" });
    return;
  }
  console.error("[ingest] request failed:", error?.message || "unknown error");
  res.status(500).json({ error: "internal_error" });
});

io.on("connection", (socket) => {
  console.log(`[io] client connected (${socket.id})`);
  socket.on("disconnect", () => console.log(`[io] client disconnected (${socket.id})`));
});

async function ensureTelemetryColumns() {
  const columns = {
    battery_soh: "DOUBLE NULL",
    motor_temp: "DOUBLE NULL",
    motor_vibration: "DOUBLE NULL",
    power_consumption: "DOUBLE NULL",
    component_health: "DOUBLE NULL",
    failure_probability: "DOUBLE NULL",
    rul_score: "DOUBLE NULL",
    maintenance_type: "VARCHAR(32) NULL",
    acceleration_mps2: "DOUBLE NULL",
    can_anomaly_score: "DOUBLE NULL",
  };
  const [existing] = await pool.query(
    "SELECT COLUMN_NAME FROM information_schema.COLUMNS WHERE TABLE_SCHEMA = ? AND TABLE_NAME = 'vehicle_telemetry'",
    [databaseUrl.pathname.slice(1)],
  );
  const known = new Set(existing.map((column) => column.COLUMN_NAME));
  if (!known.has("id")) {
    await pool.query("ALTER TABLE vehicle_telemetry ADD COLUMN id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY FIRST");
  }
  for (const [name, type] of Object.entries(columns)) {
    if (!known.has(name)) await pool.query(`ALTER TABLE vehicle_telemetry ADD COLUMN ${name} ${type}`);
  }
}

async function start() {
  let error;
  for (let attempt = 0; attempt < 30; attempt += 1) {
    try {
      await ensureTelemetryColumns();
      error = null;
      break;
    } catch (caught) {
      error = caught;
      await new Promise((resolve) => setTimeout(resolve, 2000));
    }
  }
  if (error) throw error;
  httpServer.listen(PORT, () => {
    const maskedDb = `${databaseUrl.protocol}//${databaseUrl.hostname}:${databaseUrl.port}${databaseUrl.pathname}`;
    console.log(`[ingest] listening on :${PORT} (db=${maskedDb})`);
  });
}

start().catch((error) => {
  console.error("[ingest] cannot start without the telemetry schema:", error.message);
  process.exitCode = 1;
});

process.on("SIGINT", () => {
  console.log("[ingest] shutting down");
  httpServer.close(() => pool.end().then(() => process.exit(0)));
});
