// AURA ingestion service - Express + Socket.io + TimescaleDB.
//
// Accepts telemetry samples from the simulator over HTTP POST /ingest,
// inserts into the TimescaleDB hypertable, and broadcasts each sample
// to connected dashboard clients over Socket.io.

import express from "express";
import fs from "node:fs";
import { createServer } from "node:http";
import path from "node:path";
import { Server } from "socket.io";
import pg from "pg";

const { Pool } = pg;

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
const DATABASE_URL =
  process.env.DATABASE_URL ?? "postgres://aura:aura_dev_pw@localhost:5433/aura";

const pool = new Pool({
  connectionString: DATABASE_URL,
  connectionTimeoutMillis: 2000,
  max: 5,
});

const app = express();
app.use(express.json({ limit: "64kb" }));

const httpServer = createServer(app);
const io = new Server(httpServer, {
  cors: { origin: "*", methods: ["GET", "POST"] },
});

app.get("/health", (_req, res) => {
  res.json({ status: "ok", ts: new Date().toISOString() });
});

app.get("/healthz", async (_req, res) => {
  try {
    const r = await pool.query("SELECT 1");
    res.json({ status: "ok", db: !!r.rowCount });
  } catch (e) {
    res.status(500).json({ status: "down", error: String(e) });
  }
});

function validateTelemetry(body, ip) {
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

  const time = new Date(body.time);
  if (Number.isNaN(time.getTime())) return null;

  const row = {
    time,
    vehicle_id: String(body.vehicle_id).slice(0, 64),
    pack_voltage: Number(body.pack_voltage),
    pack_current: Number(body.pack_current),
    soc_reported: Number(body.soc_reported),
    cell_temp_max: Number(body.cell_temp_max),
    speed_kmh: Number(body.speed_kmh),
    odometer_km: Number(body.odometer_km),
    latitude: Number(body.latitude),
    longitude: Number(body.longitude),
    _sourceIp: ip,
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
  ];
  if (numericKeys.some((key) => !Number.isFinite(row[key]))) return null;
  return row;
}

const INSERT_SQL = `
  INSERT INTO vehicle_telemetry
    (time, vehicle_id, pack_voltage, pack_current, soc_reported,
     cell_temp_max, speed_kmh, odometer_km, latitude, longitude)
  VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
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
  };
}

app.post("/ingest", async (req, res) => {
  const row = validateTelemetry(req.body ?? {}, req.ip);
  if (!row) {
    res.status(400).json({ error: "invalid telemetry payload" });
    return;
  }

  try {
    await pool.query(INSERT_SQL, [
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
    ]);
    io.emit("telemetry:new", socketPayload(row));
    res.status(201).json({ ok: true });
  } catch (e) {
    console.error("[ingest] insert failed:", e.message);
    io.emit("telemetry:new", socketPayload(row));
    res.status(503).json({ error: "db_unavailable", detail: e.message });
  }
});

io.on("connection", (socket) => {
  console.log(`[io] client connected (${socket.id})`);
  socket.on("disconnect", () => console.log(`[io] client disconnected (${socket.id})`));
});

httpServer.listen(PORT, () => {
  const maskedDb = DATABASE_URL.replace(/:[^:@/]+@/, ":***@");
  console.log(`[ingest] listening on :${PORT} (db=${maskedDb})`);
});

process.on("SIGINT", () => {
  console.log("[ingest] shutting down");
  httpServer.close(() => pool.end().then(() => process.exit(0)));
});
