import { useEffect, useMemo, useRef, useState } from "react";
import mapboxgl from "mapbox-gl";
import { io } from "socket.io-client";

const INGEST_URL = import.meta.env.VITE_INGEST_URL || "http://localhost:4000";
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN || "";
const MAP_STYLE = import.meta.env.VITE_MAP_STYLE || "mapbox://styles/mapbox/dark-v11";

const numberFmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });

function fmt(value, suffix = "") {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return `${numberFmt.format(Number(value))}${suffix}`;
}

function statusTone(inference) {
  if (!inference) return "border-zinc-300 bg-white text-zinc-700";
  if (inference.anomaly) return "border-red-300 bg-red-50 text-red-700";
  if (inference.advisory_action === "coast") return "border-amber-300 bg-amber-50 text-amber-800";
  return "border-emerald-300 bg-emerald-50 text-emerald-800";
}

function Kpi({ label, value, sub }) {
  return (
    <section className="rounded-lg border border-zinc-200 bg-white p-4 shadow-panel">
      <div className="text-xs font-semibold uppercase tracking-wide text-zinc-500">{label}</div>
      <div className="mt-2 text-2xl font-semibold text-zinc-950">{value}</div>
      {sub ? <div className="mt-1 text-sm text-zinc-500">{sub}</div> : null}
    </section>
  );
}

function MapPanel({ telemetry, path }) {
  const mapNode = useRef(null);
  const mapRef = useRef(null);
  const markerRef = useRef(null);
  const lineRef = useRef(null);
  const hasToken = MAPBOX_TOKEN && !MAPBOX_TOKEN.includes("YOUR_MAPBOX");

  useEffect(() => {
    if (!hasToken || mapRef.current || !mapNode.current) return;
    mapboxgl.accessToken = MAPBOX_TOKEN;
    const center = telemetry ? [telemetry.longitude, telemetry.latitude] : [-122.4194, 37.7749];
    mapRef.current = new mapboxgl.Map({
      container: mapNode.current,
      style: MAP_STYLE,
      center,
      zoom: 13,
      pitch: 42,
      bearing: -18,
      attributionControl: false,
    });
    mapRef.current.addControl(new mapboxgl.NavigationControl({ visualizePitch: true }), "bottom-right");
    markerRef.current = new mapboxgl.Marker({ color: "#10b981" }).setLngLat(center).addTo(mapRef.current);

    mapRef.current.on("load", () => {
      mapRef.current.addSource("vehicle-path", {
        type: "geojson",
        data: { type: "Feature", geometry: { type: "LineString", coordinates: [] } },
      });
      mapRef.current.addLayer({
        id: "vehicle-path-line",
        type: "line",
        source: "vehicle-path",
        paint: { "line-color": "#f59e0b", "line-width": 4, "line-opacity": 0.85 },
      });
      lineRef.current = mapRef.current.getSource("vehicle-path");
    });

    return () => {
      mapRef.current?.remove();
      mapRef.current = null;
      markerRef.current = null;
      lineRef.current = null;
    };
  }, [hasToken]);

  useEffect(() => {
    if (!telemetry || !mapRef.current || !markerRef.current) return;
    const next = [telemetry.longitude, telemetry.latitude];
    markerRef.current.setLngLat(next);
    mapRef.current.easeTo({ center: next, duration: 550 });
  }, [telemetry]);

  useEffect(() => {
    if (!mapRef.current?.isStyleLoaded()) return;
    const source = mapRef.current.getSource("vehicle-path");
    if (!source) return;
    source.setData({
      type: "Feature",
      geometry: {
        type: "LineString",
        coordinates: path.map((p) => [p.longitude, p.latitude]),
      },
    });
  }, [path]);

  return (
    <section className="relative min-h-[520px] overflow-hidden rounded-lg border border-zinc-200 bg-zinc-950 shadow-panel lg:min-h-0">
      {hasToken ? (
        <div ref={mapNode} className="absolute inset-0" />
      ) : (
        <div className="absolute inset-0 grid place-items-center bg-[radial-gradient(circle_at_30%_20%,#115e59,#18181b_55%)] px-6 text-center text-white">
          <div>
            <div className="text-lg font-semibold">Mapbox token missing</div>
            <div className="mt-2 max-w-sm text-sm text-zinc-200">Set VITE_MAPBOX_TOKEN in .env to render the live geospatial view.</div>
          </div>
        </div>
      )}
      <div className="pointer-events-none absolute left-4 top-4 rounded-lg border border-white/15 bg-black/55 px-4 py-3 text-white backdrop-blur">
        <div className="text-xs uppercase tracking-wide text-zinc-300">Vehicle</div>
        <div className="mt-1 text-lg font-semibold">{telemetry?.vehicle_id || "EV-AURA-001"}</div>
      </div>
    </section>
  );
}

function Sparkline({ points, field, color = "#0f766e" }) {
  const d = useMemo(() => {
    if (points.length < 2) return "";
    const values = points.map((p) => Number(p[field] || 0));
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = max - min || 1;
    return values
      .map((value, idx) => {
        const x = (idx / (values.length - 1)) * 100;
        const y = 40 - ((value - min) / span) * 34;
        return `${idx === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
      })
      .join(" ");
  }, [points, field]);

  return (
    <svg viewBox="0 0 100 44" className="h-16 w-full" preserveAspectRatio="none" aria-hidden="true">
      <path d={d} fill="none" stroke={color} strokeWidth="2.6" strokeLinecap="round" />
    </svg>
  );
}

export default function App() {
  const [telemetry, setTelemetry] = useState(null);
  const [path, setPath] = useState([]);
  const [inference, setInference] = useState(null);
  const [socketState, setSocketState] = useState("connecting");
  const [apiState, setApiState] = useState("loading");

  useEffect(() => {
    fetch(`${API_URL}/telemetry/recent?limit=120`)
      .then((res) => (res.ok ? res.json() : Promise.reject(res)))
      .then((data) => {
        setPath(data.items || []);
        if (data.items?.length) setTelemetry(data.items[data.items.length - 1]);
      })
      .catch(() => setApiState("waiting"));
  }, []);

  useEffect(() => {
    const socket = io(INGEST_URL, { transports: ["websocket", "polling"] });
    socket.on("connect", () => setSocketState("live"));
    socket.on("disconnect", () => setSocketState("offline"));
    socket.on("telemetry:new", (sample) => {
      setTelemetry(sample);
      setPath((prev) => [...prev.slice(-179), sample]);
    });
    return () => socket.close();
  }, []);

  useEffect(() => {
    let active = true;
    const load = () => {
      fetch(`${API_URL}/inference?limit=80`)
        .then((res) => (res.ok ? res.json() : Promise.reject(res)))
        .then((data) => {
          if (!active) return;
          setInference(data);
          setApiState("online");
          setTelemetry((current) => current || data.latest);
        })
        .catch(() => active && setApiState("waiting"));
    };
    load();
    const timer = setInterval(load, 3000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, []);

  const latest = inference?.latest || telemetry;
  const updated = latest?.time ? new Date(latest.time).toLocaleTimeString() : "--";

  return (
    <main className="min-h-screen bg-zinc-100 text-zinc-950">
      <header className="border-b border-zinc-200 bg-white">
        <div className="mx-auto flex max-w-[1480px] flex-wrap items-center justify-between gap-3 px-5 py-4">
          <div>
            <h1 className="text-xl font-semibold">AURA EV Intelligence</h1>
            <p className="text-sm text-zinc-500">Autonomous Unified Reasoning Architecture</p>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <span className={`rounded-full border px-3 py-1 font-medium ${socketState === "live" ? "border-emerald-300 bg-emerald-50 text-emerald-700" : "border-zinc-300 bg-white text-zinc-600"}`}>
              Ingest {socketState}
            </span>
            <span className={`rounded-full border px-3 py-1 font-medium ${apiState === "online" ? "border-teal-300 bg-teal-50 text-teal-700" : "border-amber-300 bg-amber-50 text-amber-700"}`}>
              ML {apiState}
            </span>
          </div>
        </div>
      </header>

      <div className="mx-auto grid max-w-[1480px] gap-5 px-5 py-5 lg:grid-cols-[330px_minmax(0,1fr)_360px]">
        <aside className="grid auto-rows-max gap-4">
          <Kpi label="State of charge" value={fmt(latest?.soc_reported, "%")} sub={`Updated ${updated}`} />
          <Kpi label="Estimated range" value={fmt(inference?.estimated_range_km, " km")} sub="BiLSTM forecast" />
          <Kpi label="Pack voltage" value={fmt(latest?.pack_voltage, " V")} sub={`Current ${fmt(latest?.pack_current, " A")}`} />
          <Kpi label="Cell temp max" value={fmt(latest?.cell_temp_max, " C")} sub={`Speed ${fmt(latest?.speed_kmh, " km/h")}`} />
        </aside>

        <MapPanel telemetry={latest} path={path} />

        <aside className="grid auto-rows-max gap-4">
          <section className={`rounded-lg border p-4 shadow-panel ${statusTone(inference)}`}>
            <div className="text-xs font-semibold uppercase tracking-wide opacity-70">Reasoning state</div>
            <div className="mt-2 text-2xl font-semibold">{inference?.anomaly ? "Anomaly detected" : "Nominal"}</div>
            <div className="mt-1 text-sm">Isolation score {fmt(inference?.anomaly_score)}</div>
          </section>

          <section className="rounded-lg border border-zinc-200 bg-white p-4 shadow-panel">
            <div className="flex items-center justify-between gap-4">
              <div>
                <div className="text-xs font-semibold uppercase tracking-wide text-zinc-500">DQN advisory</div>
                <div className="mt-2 text-2xl font-semibold capitalize">{inference?.advisory_action || "--"}</div>
              </div>
              <div className="text-right text-sm text-zinc-500">
                Confidence
                <div className="text-lg font-semibold text-zinc-950">{fmt((inference?.advisory_confidence || 0) * 100, "%")}</div>
              </div>
            </div>
          </section>

          <section className="rounded-lg border border-zinc-200 bg-white p-4 shadow-panel">
            <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-zinc-500">SOC trend</div>
            <Sparkline points={path} field="soc_reported" color="#0f766e" />
          </section>

          <section className="rounded-lg border border-zinc-200 bg-white p-4 shadow-panel">
            <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-zinc-500">Thermal trend</div>
            <Sparkline points={path} field="cell_temp_max" color="#dc2626" />
          </section>

          <section className="rounded-lg border border-zinc-200 bg-white p-4 text-sm shadow-panel">
            <div className="grid grid-cols-2 gap-x-4 gap-y-3">
              <div>
                <div className="text-zinc-500">Odometer</div>
                <div className="font-semibold">{fmt(latest?.odometer_km, " km")}</div>
              </div>
              <div>
                <div className="text-zinc-500">Coordinates</div>
                <div className="font-semibold">{fmt(latest?.latitude)}, {fmt(latest?.longitude)}</div>
              </div>
            </div>
          </section>
        </aside>
      </div>
    </main>
  );
}
