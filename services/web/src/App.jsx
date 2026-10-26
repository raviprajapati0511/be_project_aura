import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { io } from "socket.io-client";

const INGEST_URL = import.meta.env.VITE_INGEST_URL || "http://localhost:4000";
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN || "";
const MAP_STYLE = import.meta.env.VITE_MAP_STYLE || "mapbox://styles/mapbox/dark-v11";

const numberFmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });
const timeFmt = new Intl.DateTimeFormat("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit" });

function fmt(value, suffix = "") {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
  return `${numberFmt.format(Number(value))}${suffix}`;
}

function Icon({ name, size = 18 }) {
  const paths = {
    overview: <><rect x="3" y="3" width="7" height="7" rx="1.5" /><rect x="14" y="3" width="7" height="7" rx="1.5" /><rect x="3" y="14" width="7" height="7" rx="1.5" /><rect x="14" y="14" width="7" height="7" rx="1.5" /></>,
    vehicle: <><path d="m5 11 1.5-4.5A2 2 0 0 1 8.4 5h7.2a2 2 0 0 1 1.9 1.5L19 11" /><path d="M4 11h16v7H4zM7 18v2m10-2v2M7 14h.01M17 14h.01" /></>,
    pulse: <><path d="M3 12h4l3-8 4 16 3-8h4" /></>,
    route: <><circle cx="6" cy="18" r="2" /><circle cx="18" cy="6" r="2" /><path d="M8 18h3a3 3 0 0 0 3-3V9a3 3 0 0 1 3-3" /></>,
    settings: <><circle cx="12" cy="12" r="3" /><path d="m19.4 15 .1.1 1.4 1.1-1.4 2.4-1.7-.7a8 8 0 0 1-1.7 1l-.3 1.8h-2.8l-.3-1.8a8 8 0 0 1-1.7-1l-1.7.7-1.4-2.4 1.4-1.1a7 7 0 0 1 0-2l-1.4-1.1 1.4-2.4 1.7.7a8 8 0 0 1 1.7-1l.3-1.8h2.8l.3 1.8a8 8 0 0 1 1.7 1l1.7-.7 1.4 2.4-1.4 1.1a7 7 0 0 1 0 2Z" transform="translate(-1 -1) scale(.94)" /></>,
    bolt: <path d="m13 2-9 12h7l-1 8 10-13h-7z" />,
    arrow: <><path d="M5 12h14" /><path d="m13 6 6 6-6 6" /></>,
    pin: <><path d="M20 10c0 5-8 12-8 12S4 15 4 10a8 8 0 1 1 16 0Z" /><circle cx="12" cy="10" r="2.5" /></>,
    clock: <><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></>,
    check: <><path d="m5 12 4 4L19 6" /></>,
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name] || paths.pulse}</svg>;
}

function ServiceBadge({ label, state }) {
  const live = state === "online";
  return <span className={`service-badge ${live ? "is-live" : "is-idle"}`}><i />{label}<b>{live ? "Live" : state === "loading" ? "Connecting" : "Offline"}</b></span>;
}

function MetricCard({ icon, label, value, unit, note, accent = "teal", progress }) {
  return (
    <section className={`metric-card accent-${accent}`}>
      <div className="metric-top"><span className="metric-icon"><Icon name={icon} size={17} /></span><span className="metric-label">{label}</span><span className="metric-menu">···</span></div>
      <div className="metric-value">{value}<small>{unit}</small></div>
      <div className="metric-foot">{progress !== undefined ? <div className="soc-track"><span style={{ width: `${Math.max(0, Math.min(100, progress))}%` }} /></div> : <span className="metric-dot" />}{note}</div>
    </section>
  );
}

function SparkChart({ points, field, color = "#20b89a", label }) {
  const values = points.map((point) => Number(point[field])).filter(Number.isFinite);
  const { line, area } = useMemo(() => {
    if (values.length < 2) return { line: "", area: "" };
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = max - min || 1;
    const coords = values.map((value, index) => {
      const x = 4 + (index / (values.length - 1)) * 92;
      const y = 54 - ((value - min) / span) * 42;
      return [x, y];
    });
    const path = coords.map(([x, y], index) => `${index ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
    return { line: path, area: `${path} L96,62 L4,62 Z` };
  }, [values.join(",")]);
  const latest = values.at(-1);
  return (
    <div className="chart-wrap">
      <div className="chart-legend"><span><i style={{ background: color }} />{label}</span><strong>{fmt(latest)}</strong></div>
      <svg className="trend-chart" viewBox="0 0 100 66" preserveAspectRatio="none" role="img" aria-label={`${label} trend`}>
        <defs><linearGradient id={`fill-${field}`} x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stopColor={color} stopOpacity=".18" /><stop offset="100%" stopColor={color} stopOpacity="0" /></linearGradient></defs>
        {[12, 29, 46, 62].map((y) => <line key={y} x1="4" x2="96" y1={y} y2={y} className="chart-grid" />)}
        {area && <path d={area} fill={`url(#fill-${field})`} />}
        {line && <path d={line} fill="none" stroke={color} strokeWidth="1.7" vectorEffect="non-scaling-stroke" strokeLinecap="round" strokeLinejoin="round" />}
      </svg>
      <div className="chart-axis"><span>Oldest</span><span>Latest</span></div>
    </div>
  );
}

function MapPanel({ telemetry, path }) {
  const nodeRef = useRef(null);
  const mapRef = useRef(null);
  const markerRef = useRef(null);
  const pathRef = useRef(path);
  const hasToken = Boolean(MAPBOX_TOKEN && !MAPBOX_TOKEN.includes("YOUR_MAPBOX"));
  const coordinates = telemetry && Number.isFinite(Number(telemetry.latitude)) && Number.isFinite(Number(telemetry.longitude));

  useEffect(() => { pathRef.current = path; }, [path]);

  useEffect(() => {
    if (!hasToken || !nodeRef.current || mapRef.current) return undefined;
    let cancelled = false;
    let map;
    import("mapbox-gl").then(({ default: mapboxgl }) => {
      if (cancelled || !nodeRef.current) return;
      mapboxgl.accessToken = MAPBOX_TOKEN;
      const center = coordinates ? [Number(telemetry.longitude), Number(telemetry.latitude)] : [-122.4194, 37.7749];
      map = new mapboxgl.Map({ container: nodeRef.current, style: MAP_STYLE, center, zoom: 13, attributionControl: false });
      mapRef.current = map;
      map.addControl(new mapboxgl.NavigationControl({ showCompass: false }), "bottom-right");
      const marker = new mapboxgl.Marker({ color: "#22c9a2" }).setLngLat(center).addTo(map);
      markerRef.current = marker;
      map.on("load", () => {
        map.addSource("telemetry-path", { type: "geojson", data: { type: "Feature", geometry: { type: "LineString", coordinates: [] } } });
        map.addLayer({ id: "telemetry-path-line", type: "line", source: "telemetry-path", paint: { "line-color": "#22c9a2", "line-width": 3, "line-opacity": 0.8 } });
        map.getSource("telemetry-path").setData({ type: "Feature", geometry: { type: "LineString", coordinates: pathRef.current.map((p) => [Number(p.longitude), Number(p.latitude)]) } });
      });
    }).catch((error) => console.error("Mapbox failed to load:", error));
    return () => { cancelled = true; map?.remove(); mapRef.current = null; markerRef.current = null; };
  }, [hasToken]);

  useEffect(() => {
    if (!coordinates || !mapRef.current || !markerRef.current) return;
    const point = [Number(telemetry.longitude), Number(telemetry.latitude)];
    markerRef.current.setLngLat(point);
    mapRef.current.easeTo({ center: point, duration: 500 });
  }, [telemetry?.latitude, telemetry?.longitude, coordinates]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map?.isStyleLoaded()) return;
    const source = map.getSource("telemetry-path");
    if (!source) return;
    source.setData({ type: "Feature", geometry: { type: "LineString", coordinates: path.filter((p) => Number.isFinite(Number(p.longitude)) && Number.isFinite(Number(p.latitude))).map((p) => [Number(p.longitude), Number(p.latitude)]) } });
  }, [path]);

  return (
    <section className="panel map-panel">
      <div className="panel-heading map-heading"><div><div className="eyebrow">LIVE VEHICLE POSITION</div><h2>Route overview</h2></div><span className="map-live"><i />Tracking</span></div>
      <div className="map-canvas">
        {hasToken ? <div ref={nodeRef} className="mapbox-canvas" /> : (
          <div className="map-placeholder">
            <div className="map-street street-a" /><div className="map-street street-b" /><div className="map-street street-c" /><div className="map-block block-a" /><div className="map-block block-b" /><div className="map-block block-c" />
            <div className="map-route"><span className="route-point point-one" /><span className="route-point point-two" /><span className="route-point point-three" /></div>
            <div className="map-pin"><span><Icon name="vehicle" size={19} /></span></div>
            <div className="map-provider">SCHEMATIC MAP · MAPBOX TOKEN REQUIRED</div>
          </div>
        )}
        <div className="map-coordinate"><Icon name="pin" size={14} />{coordinates ? `${Number(telemetry.latitude).toFixed(4)}, ${Number(telemetry.longitude).toFixed(4)}` : "Awaiting GPS signal"}</div>
      </div>
      <div className="map-footer"><div><span className="footer-label">VEHICLE</span><strong>{telemetry?.vehicle_id || "No vehicle selected"}</strong></div><div><span className="footer-label">SPEED</span><strong>{fmt(telemetry?.speed_kmh, " km/h")}</strong></div><div><span className="footer-label">ODOMETER</span><strong>{fmt(telemetry?.odometer_km, " km")}</strong></div></div>
    </section>
  );
}

function App() {
  const [vehicles, setVehicles] = useState([]);
  const [vehicleId, setVehicleId] = useState("");
  const [telemetry, setTelemetry] = useState(null);
  const [path, setPath] = useState([]);
  const [inference, setInference] = useState(null);
  const [socketState, setSocketState] = useState("loading");
  const [apiState, setApiState] = useState("loading");
  const [lastSeen, setLastSeen] = useState(null);
  const [fleetSummary, setFleetSummary] = useState(null);
  const [researchReport, setResearchReport] = useState(null);
  const [securitySummary, setSecuritySummary] = useState(null);
  const [chargingForecast, setChargingForecast] = useState(null);
  const [chargingForecastState, setChargingForecastState] = useState("idle");
  const [streamPaused, setStreamPaused] = useState(false);
  const [activitySearch, setActivitySearch] = useState("");
  const vehicleIdRef = useRef(vehicleId);
  const streamPausedRef = useRef(streamPaused);
  useEffect(() => { vehicleIdRef.current = vehicleId; }, [vehicleId]);
  useEffect(() => { streamPausedRef.current = streamPaused; }, [streamPaused]);

  const vehicleQuery = vehicleId ? `?vehicle_id=${encodeURIComponent(vehicleId)}` : "";
  const acceptSample = useCallback((sample) => {
    if (streamPausedRef.current) return;
    if (vehicleIdRef.current && sample.vehicle_id !== vehicleIdRef.current) return;
    setTelemetry(sample);
    setLastSeen(new Date());
    setPath((previous) => [...previous.slice(-119), sample]);
  }, []);

  useEffect(() => {
    let active = true;
    const load = async () => {
      try {
        const response = await fetch(`${API_URL}/fleet/summary`);
        if (!response.ok) return;
        const data = await response.json();
        if (active) setFleetSummary(data);
      } catch { /* Connection banner and reconnect loop cover unavailable services. */ }
    };
    load();
    const timer = setInterval(load, 10000);
    return () => { active = false; clearInterval(timer); };
  }, []);

  useEffect(() => {
    let active = true;
    fetch(`${API_URL}/research/evaluation`)
      .then((response) => response.ok ? response.json() : null)
      .then((data) => active && data && setResearchReport(data))
      .catch(() => {});
    return () => { active = false; };
  }, []);

  useEffect(() => {
    let active = true;
    const load = async () => {
      try {
        const response = await fetch(`${INGEST_URL}/security/summary`);
        if (response.ok) {
          const data = await response.json();
          if (active) setSecuritySummary(data);
        }
      } catch { /* Ingestion service badge reports the connection state. */ }
    };
    load();
    const timer = setInterval(load, 10000);
    return () => { active = false; clearInterval(timer); };
  }, []);

  useEffect(() => {
    let active = true;
    const loadVehicles = async () => {
      try {
        const response = await fetch(`${API_URL}/vehicles`);
        if (!response.ok) throw new Error("vehicle list unavailable");
        const data = await response.json();
        if (!active) return;
        const list = data.vehicles || [];
        setVehicles(list);
        setVehicleId((current) => current && list.includes(current) ? current : (list[0] || ""));
        setApiState("online");
      } catch {
        if (active) setApiState("offline");
      }
    };
    loadVehicles();
    const timer = setInterval(loadVehicles, 5000);
    return () => { active = false; clearInterval(timer); };
  }, []);

  useEffect(() => {
    const socket = io(INGEST_URL, { transports: ["websocket", "polling"], reconnection: true, reconnectionDelayMax: 5000 });
    socket.on("connect", () => setSocketState("online"));
    socket.on("disconnect", () => setSocketState("offline"));
    socket.on("connect_error", () => setSocketState("offline"));
    socket.on("telemetry:new", acceptSample);
    return () => socket.close();
  }, [acceptSample]);

  useEffect(() => {
    let active = true;
    if (!vehicleId) return () => { active = false; };
    fetch(`${API_URL}/telemetry/recent?limit=120&vehicle_id=${encodeURIComponent(vehicleId)}`)
      .then((response) => response.ok ? response.json() : Promise.reject(response))
      .then((data) => {
        if (!active) return;
        const items = data.items || [];
        setPath(items);
        if (items.length) { setTelemetry(items.at(-1)); setLastSeen(new Date(items.at(-1).time)); }
      })
      .catch(() => active && setApiState("offline"));
    return () => { active = false; };
  }, [vehicleId]);

  useEffect(() => {
    if (!vehicleId) return undefined;
    let active = true;
    let pending = false;
    const load = async () => {
      if (pending || streamPausedRef.current) return;
      pending = true;
      try {
        const response = await fetch(`${API_URL}/inference${vehicleQuery ? `${vehicleQuery}&limit=120` : "?limit=120"}`);
        if (!response.ok) throw new Error("inference unavailable");
        const data = await response.json();
        if (!active) return;
        setInference(data);
        setTelemetry((current) => current || data.latest);
        setLastSeen((current) => {
          const observed = new Date(data.latest.time);
          return current && current > observed ? current : observed;
        });
        setApiState("online");
      } catch {
        if (active) setApiState("offline");
      } finally { pending = false; }
    };
    load();
    const timer = setInterval(load, 5000);
    return () => { active = false; clearInterval(timer); };
  }, [vehicleId, vehicleQuery]);

  const inferenceTime = inference?.latest?.time ? new Date(inference.latest.time) : null;
  const telemetryTime = telemetry?.time ? new Date(telemetry.time) : null;
  const latest = telemetry && telemetryTime > inferenceTime ? telemetry : (inference?.latest || telemetry);
  const signals = inference?.signals || [];
  const alert = inference?.reasoning_status === "investigate";
  const updated = latest?.time ? timeFmt.format(new Date(latest.time)) : "Waiting for data";
  const visibleSamples = useMemo(() => {
    const query = activitySearch.trim().toLowerCase();
    if (!query) return path;
    return path.filter((row) => [row.time, row.vehicle_id, row.maintenance_type].some((value) => String(value || "").toLowerCase().includes(query)));
  }, [path, activitySearch]);
  const exportTelemetry = () => {
    const fields = ["time", "vehicle_id", "soc_reported", "pack_voltage", "pack_current", "cell_temp_max", "motor_temp", "speed_kmh", "battery_soh", "component_health", "failure_probability", "rul_score", "maintenance_type"];
    const csv = [fields.join(","), ...path.map((row) => fields.map((key) => `"${String(row[key] ?? "").replaceAll('"', '""')}"`).join(","))].join("\r\n");
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
    link.download = `${vehicleId || "aura-fleet"}-telemetry.csv`;
    link.click();
    URL.revokeObjectURL(link.href);
  };

  const forecastSession = async () => {
    const socStart = Math.max(0, Math.min(100, Number(inference?.estimated_soc_pct ?? latest?.soc_reported ?? 30)));
    const socTarget = Math.max(socStart, Math.min(90, Math.ceil(socStart / 10) * 10 + 10));
    const params = new URLSearchParams({ soc_start_pct: String(socStart), soc_target_pct: String(socTarget) });
    setChargingForecastState("loading");
    try {
      const response = await fetch(`${API_URL}/charging/forecast?${params}`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Forecast model unavailable");
      setChargingForecast(data.forecast);
      setChargingForecastState("ready");
    } catch (error) {
      setChargingForecast(null);
      setChargingForecastState(error.message || "Forecast unavailable");
    }
  };
  const researchModules = researchReport?.modules || {};
  const scorecard = [
    ["SoC CNN-BiLSTM", `MAE ${fmt(researchModules.soc_range_cnn_bilstm?.soc?.mae)} pp · RMSE ${fmt(researchModules.soc_range_cnn_bilstm?.soc?.rmse)} pp`, "Chronological holdout; compared with noisy sensor baseline."],
    ["Range CNN-BiLSTM", `MAE ${fmt(researchModules.soc_range_cnn_bilstm?.range_proxy?.mae)} km · RMSE ${fmt(researchModules.soc_range_cnn_bilstm?.range_proxy?.rmse)} km`, "Derived range proxy, not a measured range label."],
    ["Isolation Forest", `P ${fmt(researchModules.isolation_forest_vs_failure_label_proxy?.precision)} · R ${fmt(researchModules.isolation_forest_vs_failure_label_proxy?.recall)} · F1 ${fmt(researchModules.isolation_forest_vs_failure_label_proxy?.f1)} · FPR ${fmt(researchModules.isolation_forest_vs_failure_label_proxy?.false_positive_rate)}`, "Failure label used only as an anomaly proxy."],
    ["Threshold baseline", `P ${fmt(researchModules.threshold_baseline_vs_failure_label_proxy?.precision)} · R ${fmt(researchModules.threshold_baseline_vs_failure_label_proxy?.recall)} · F1 ${fmt(researchModules.threshold_baseline_vs_failure_label_proxy?.f1)} · FPR ${fmt(researchModules.threshold_baseline_vs_failure_label_proxy?.false_positive_rate)}`, "Same proxy labels as the anomaly model."],
    ["Cross-domain fusion", `P ${fmt(researchModules.cross_domain_fusion_ablation_vs_failure_label_proxy?.precision)} · R ${fmt(researchModules.cross_domain_fusion_ablation_vs_failure_label_proxy?.recall)} · F1 ${fmt(researchModules.cross_domain_fusion_ablation_vs_failure_label_proxy?.f1)} · FPR ${fmt(researchModules.cross_domain_fusion_ablation_vs_failure_label_proxy?.false_positive_rate)}`, "Ablation compared with Isolation Forest and threshold rules."],
    ["Failure classifier", `P ${fmt(researchModules.failure_classifier?.precision)} · R ${fmt(researchModules.failure_classifier?.recall)} · F1 ${fmt(researchModules.failure_classifier?.f1)} · FPR ${fmt(researchModules.failure_classifier?.false_positive_rate)}`, "Published Kaggle failure labels; chronological holdout."],
    ["RUL regressor", `MAE ${fmt(researchModules.rul_regressor?.mae)} · RMSE ${fmt(researchModules.rul_regressor?.rmse)}`, "Dataset score; source does not specify units."],
    ["Kaggle session forecaster", `Energy MAE ${fmt(researchModules.charging_session_forecaster?.["Energy Consumed (kWh)"]?.mae)} kWh | Duration MAE ${fmt(researchModules.charging_session_forecaster?.["Charging Duration (hours)"]?.mae)} h`, "Separate Kaggle session dataset; weak predictive signal on the chronological 80/20 holdout."],
    ["CAN intrusion classifier", `Macro F1 ${fmt(researchModules.can_intrusion_detection?.macro_f1)} | Binary intrusion F1 ${fmt(researchModules.can_intrusion_detection?.binary_intrusion_f1)}`, "HCRL windowed CAN classes; chronological capture holdout, not unseen-vehicle validation."],
    ["DQN (raw)", `Cost ${fmt(researchModules.smart_charging_dqn?.raw_dqn_mean_cost)} USD · target ${fmt(Number(researchModules.smart_charging_dqn?.raw_dqn_target_completion_rate) * 100)}%`, "Unsupervised policy proposal before the target/peak supervisor."],
    ["DQN + safety shield", `Cost ${fmt(researchModules.smart_charging_dqn?.dqn_mean_cost)} USD · peak ${fmt(researchModules.smart_charging_dqn?.dqn_mean_peak_load)} · violations ${fmt(researchModules.smart_charging_dqn?.dqn_mean_constraint_violations)} · target ${fmt(Number(researchModules.smart_charging_dqn?.dqn_target_completion_rate) * 100)}%`, "Synthetic charging episodes; compared against the rule baseline."],
    ["Rule charging baseline", `Cost ${fmt(researchModules.smart_charging_dqn?.rule_baseline_mean_cost)} USD · peak ${fmt(researchModules.smart_charging_dqn?.rule_baseline_mean_peak_load)} · violations ${fmt(researchModules.smart_charging_dqn?.rule_baseline_mean_constraint_violations)} · target ${fmt(Number(researchModules.smart_charging_dqn?.rule_baseline_target_completion_rate) * 100)}%`, "Synthetic tariff and grid environment."],
  ];

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><div className="brand-mark"><span /><span /><span /></div><div><strong>AURA</strong><small>EV INTELLIGENCE</small></div></div>
        <div className="workspace-label">WORKSPACE</div>
        <nav className="nav-list" aria-label="Main navigation"><a className="nav-item active" href="#overview"><Icon name="overview" />Overview</a><a className="nav-item" href="#fleet"><Icon name="vehicle" />Fleet<span className="nav-count">{vehicles.length || "—"}</span></a><a className="nav-item" href="#charging"><Icon name="bolt" />Charging</a><a className="nav-item" href="#insights"><Icon name="pulse" />Intelligence</a><a className="nav-item" href="#research"><Icon name="route" />Research</a><a className="nav-item" href="#security"><Icon name="settings" />Security</a></nav>
        <div className="sidebar-bottom"><div className="system-card"><div className="system-orb"><Icon name="bolt" size={17} /></div><div><strong>System status</strong><span>{apiState === "online" && socketState === "online" ? "All services operational" : "Waiting for services"}</span></div><i className={apiState === "online" && socketState === "online" ? "status-green" : "status-amber"} /></div><div className="profile"><div className="profile-avatar">A</div><div><strong>AURA Operator</strong><span>Local workspace</span></div><span className="profile-menu">···</span></div></div>
      </aside>

      <main className="main-content" id="overview">
        <header className="topbar"><div className="breadcrumb">Workspace <span>/</span> <strong>Overview</strong></div><div className="topbar-right"><ServiceBadge label="Ingestion" state={socketState} /><ServiceBadge label="AI engine" state={apiState} /><div className="top-divider" /><button className="avatar-button" aria-label="Account">A</button></div></header>
        <div className="page-wrap">
          <section className="page-title-row"><div><div className="eyebrow">MONITORING CENTER <span className="title-separator">/</span> FLEET OPERATIONS <span className="prototype-tag">RESEARCH PROTOTYPE</span></div><h1>Good day, Operator <span className="wave">✳</span></h1><p className="page-subtitle">A real-time view of vehicle health, energy, and system intelligence.</p></div><div className="title-actions"><label className="vehicle-select"><Icon name="vehicle" size={16} /><select value={vehicleId} onChange={(event) => { setVehicleId(event.target.value); setPath([]); setTelemetry(null); setInference(null); setLastSeen(null); }} aria-label="Select vehicle"><option value="">Select vehicle</option>{vehicles.map((vehicle) => <option key={vehicle} value={vehicle}>{vehicle}</option>)}</select><span className="select-chevron">⌄</span></label><div className="updated-label"><span>LAST UPDATED</span><strong>{updated}</strong></div></div></section>

          {apiState === "offline" && <div className="connection-banner"><span className="banner-icon">!</span><div><strong>Telemetry services are unavailable</strong><span>Check that the database and API services are running. The dashboard will reconnect automatically.</span></div></div>}

          <section className="metric-grid" id="vehicle" aria-label="Vehicle metrics">
            <MetricCard icon="bolt" label="State of charge" value={fmt(latest?.soc_reported)} unit="%" note="Available battery capacity" accent="teal" progress={latest?.soc_reported} />
            <MetricCard icon="route" label="Estimated range" value={fmt(inference?.estimated_range_km)} unit="km" note="SoC and battery health estimate" accent="blue" />
            <MetricCard icon="pulse" label="Pack voltage" value={fmt(latest?.pack_voltage)} unit="V" note={`Current draw ${fmt(latest?.pack_current, " A")}`} accent="violet" />
            <MetricCard icon="vehicle" label="Cell temperature" value={fmt(latest?.cell_temp_max)} unit="°C" note={`Vehicle speed ${fmt(latest?.speed_kmh, " km/h")}`} accent={Number(latest?.cell_temp_max) >= 48 ? "red" : "amber"} />
            <MetricCard icon="pulse" label="Battery health" value={fmt(latest?.battery_soh != null ? latest.battery_soh * 100 : null)} unit="%" note="Dataset state of health" accent="teal" />
            <MetricCard icon="pulse" label="Failure risk" value={fmt(Number.isFinite(Number(inference?.failure_probability)) ? inference.failure_probability * 100 : null)} unit="%" note={inference?.model_status === "full" ? "Trained model stack estimate" : "Fallback or partial model stack"} accent={Number(inference?.failure_probability) >= 0.35 ? "red" : "blue"} />
            <MetricCard icon="clock" label="Remaining useful life" value={fmt(inference?.remaining_useful_life)} unit="" note="Dataset score; unit not specified" accent="violet" />
            <MetricCard icon="vehicle" label="Motor temperature" value={fmt(latest?.motor_temp)} unit="°C" note={`Vibration ${fmt(latest?.motor_vibration)}`} accent={Number(latest?.motor_temp) >= 85 ? "red" : "amber"} />
          </section>

          <section className="panel fleet-panel" id="fleet">
            <div className="panel-heading"><div><div className="eyebrow">FLEET OVERVIEW</div><h2>Vehicle status</h2></div><span className="window-tag"><i />{fleetSummary?.vehicle_count ?? "—"} vehicles reporting</span></div>
            <div className="fleet-kpis"><div><span>VEHICLES</span><strong>{fleetSummary?.vehicle_count ?? "—"}</strong></div><div><span>NEEDS REVIEW</span><strong className={fleetSummary?.at_risk_count ? "risk-number" : ""}>{fleetSummary?.at_risk_count ?? "—"}</strong></div><div><span>FLEET AVG SOC</span><strong>{fmt(fleetSummary?.mean_soc_pct, "%")}</strong></div><div><span>FLEET AVG SOH</span><strong>{fmt(fleetSummary?.mean_battery_health_pct, "%")}</strong></div></div>
            <div className="table-scroll"><table><thead><tr><th>VEHICLE</th><th>STATE OF CHARGE</th><th>BATTERY HEALTH</th><th>FAILURE RISK</th><th>SPEED</th><th>STATUS</th></tr></thead><tbody>{(fleetSummary?.vehicles || []).map((vehicle) => <tr key={vehicle.vehicle_id} className="fleet-row" onClick={() => { setVehicleId(vehicle.vehicle_id); setPath([]); setTelemetry(vehicle); setInference(null); }}><td><strong>{vehicle.vehicle_id}</strong></td><td>{fmt(vehicle.soc_reported, "%")}</td><td>{fmt(vehicle.battery_soh != null ? vehicle.battery_soh * 100 : null, "%")}</td><td>{fmt(vehicle.failure_probability * 100, "%")}</td><td>{fmt(vehicle.speed_kmh, " km/h")}</td><td><span className={`row-status ${vehicle.reasoning_status === "investigate" ? "needs-review" : ""}`}><i />{vehicle.reasoning_status === "investigate" ? "Review" : "Nominal"}</span></td></tr>)}</tbody></table>{!fleetSummary?.vehicles?.length && <div className="empty-stream">Fleet data will appear here after telemetry arrives.</div>}</div>
          </section>

          <section className="primary-grid">
            <div id="route"><MapPanel telemetry={latest} path={path} /></div>
            <section className={`panel intelligence-panel ${alert ? "has-alert" : ""}`} id="insights">
              <div className="panel-heading"><div><div className="eyebrow">AURA REASONING</div><h2>System intelligence</h2></div><span className={`health-pill ${!inference ? "pending" : alert ? "warning" : "healthy"}`}><i />{!inference ? "Awaiting data" : alert ? "Review advised" : "Nominal"}</span></div>
              <div className="intelligence-summary"><div className={`intelligence-icon ${!inference ? "pending" : alert ? "warning" : "healthy"}`}><Icon name={!inference ? "clock" : alert ? "pulse" : "check"} size={20} /></div><div><strong>{!inference ? "Waiting for vehicle telemetry" : alert ? "Signal needs attention" : "Vehicle operating normally"}</strong><p>{inference?.reasoning_summary || "AI evaluation will appear when the selected vehicle sends data."}</p></div></div>
              {inference?.reasoning && <div className="reasoning-detail"><div><span>WORKING HYPOTHESIS</span><strong>{inference.reasoning.label.replaceAll("_", " ")}</strong><small>{inference.reasoning.confidence ? `${Math.round(inference.reasoning.confidence * 100)}% heuristic confidence` : "No security confidence"}</small></div><p>{inference.reasoning.explanation || inference.reasoning.limitations}</p><small>Power residual z-score {fmt(inference.reasoning.evidence?.power_residual_z)} · CAN detector {inference.reasoning.evidence?.can_detector_available ? `connected (${fmt(inference.reasoning.evidence?.can_anomaly_score)})` : "not connected"}</small></div>}
              <div className="reasoning-divider" />
              <div className="section-caption">CROSS-DOMAIN SIGNALS <span>{signals.length ? `${signals.length} active` : inference ? "No active signals" : "Awaiting data"}</span></div>
              {signals.length ? <div className="signal-list">{signals.slice(0, 3).map((signal) => <div className="signal-row" key={signal.code}><span className={`signal-dot ${signal.severity}`} /><div><strong>{signal.code.replaceAll("_", " ")}</strong><p>{signal.message}</p></div><span className={`severity-tag ${signal.severity}`}>{signal.severity}</span></div>)}</div> : inference ? <div className="all-clear"><span><Icon name="check" size={15} /></span><div><strong>No configured telemetry alerts</strong><small>Electrical · Thermal · Driving</small></div></div> : <div className="all-clear waiting"><span><Icon name="clock" size={15} /></span><div><strong>No assessment yet</strong><small>Connect a vehicle to start monitoring</small></div></div>}
              <div className="model-foot"><span>ANOMALY MODEL SCORE</span><strong>{fmt(inference?.anomaly_score)}</strong><span className="model-foot-divider" /><span>CHARGING POLICY</span><strong className="capitalize">{inference?.charging?.controller || "—"}</strong></div>
            </section>
          </section>

          <section className="charging-grid" id="charging">
            <section className="panel charging-panel"><div className="panel-heading"><div><div className="eyebrow">REINFORCEMENT LEARNING</div><h2>Smart charging plan</h2></div><span className={`health-pill ${inference?.charging?.controller?.startsWith("DQN") ? "healthy" : "pending"}`}><i />{inference?.charging?.controller || "Waiting for policy"}</span></div>
              <div className="charge-action"><span className="action-label">POLICY RECOMMENDATION</span><strong>{inference?.charging?.recommended_action || "Awaiting telemetry"}</strong><p>Target charge {fmt(Number(inference?.charging?.target_soc) * 100, "%")} · Charge power {fmt(inference?.charging?.power_kw, " kW")}</p></div>
              <div className="charge-context"><div><span>DEMO TARIFF</span><strong>{fmt(inference?.charging?.price, " USD/kWh")}</strong></div><div><span>GRID LOAD</span><strong>{fmt(Number(inference?.charging?.grid_load) * 100, "%")}</strong></div><div><span>RENEWABLE SHARE</span><strong>{fmt(Number(inference?.charging?.renewable) * 100, "%")}</strong></div></div>
              <div className="charge-forecast-control"><button className="utility-button" onClick={forecastSession} disabled={chargingForecastState === "loading"}>{chargingForecastState === "loading" ? "Estimating…" : "Forecast session from Kaggle data"}</button>{chargingForecast && <span>Estimated {fmt(chargingForecast.estimated_energy_kwh, " kWh")} · {fmt(chargingForecast.estimated_duration_hours, " h")}</span>}{chargingForecastState !== "idle" && chargingForecastState !== "ready" && chargingForecastState !== "loading" && <span>{chargingForecastState}</span>}</div>
              {chargingForecast && <div className="module-note">Illustrative estimate only: this dataset has weak chronological holdout accuracy.</div>}
              {inference?.charging?.safety_restricted && <div className="module-note">Policy proposed {inference.charging.proposed_action}; the telemetry safety filter restricted it.</div>}
              <div className="module-note">Research simulator only. Tariff and grid conditions are generated for the demo; this does not control a real charger.</div>
            </section>
            <section className="panel cyber-panel" id="security"><div className="panel-heading"><div><div className="eyebrow">COMMUNICATION MONITORING</div><h2>Security and data quality</h2></div><span className="health-pill pending"><i />Ingress scope</span></div>
              <div className="security-stats"><div><span>REJECTED PAYLOADS</span><strong>{securitySummary?.rejected_payloads ?? "—"}</strong></div><div><span>RATE-LIMITED</span><strong>{securitySummary?.rate_limited ?? "—"}</strong></div><div><span>MALFORMED JSON</span><strong>{securitySummary?.malformed_json ?? "—"}</strong></div></div>
              <div className="module-note">This monitors the local ingestion API and payload validation. The dataset contains no CAN frames, network traffic, or labeled attacks, so it cannot identify cyberattacks. Telemetry anomalies are investigation signals, not cyber verdicts.</div>
            </section>
          </section>

          <section className="secondary-grid">
            <section className="panel trend-panel"><div className="panel-heading"><div><div className="eyebrow">LIVE TELEMETRY</div><h2>Energy & thermal trends</h2></div><span className="window-tag"><i />Live · 120 samples</span></div><div className="trend-grid"><div className="trend-card"><SparkChart points={path} field="soc_reported" label="State of charge" color="#22b89a" /><div className="trend-range">{path.length ? `${path.length} samples` : "No samples yet"}</div></div><div className="trend-card"><SparkChart points={path} field="cell_temp_max" label="Cell temperature" color="#e89b4b" /><div className="trend-range">Peak {fmt(Math.max(...path.map((p) => Number(p.cell_temp_max) || 0)), " °C")}</div></div><div className="trend-card"><SparkChart points={path} field="pack_current" label="Pack current" color="#7793f4" /><div className="trend-range">Latest {fmt(latest?.pack_current, " A")}</div></div></div></section>
            <section className="panel advisory-panel"><div className="panel-heading"><div><div className="eyebrow">CHARGING SUPPORT</div><h2>Smart charging advisory</h2></div><span className="advisory-mark"><Icon name="bolt" size={16} /></span></div><div className="advisory-action"><span className="action-label">RECOMMENDED ACTION</span><strong className="capitalize">{inference?.advisory_action || "Awaiting analysis"}</strong><p>{inference?.advisory_action === "defer" ? "Wait for a lower tariff or grid load if the target charge and departure schedule allow." : inference?.advisory_action ? "Charge at the recommended power while respecting the demo battery and grid constraints." : "Charging guidance will appear when the selected vehicle sends data."}</p></div><div className="advisory-meta"><span>Controller</span><strong>{inference?.charging?.controller || "Awaiting model"}</strong></div><div className="advisory-note"><Icon name="pulse" size={14} />Research decision support · Does not control a real charger</div></section>
          </section>

          <section className="panel activity-panel" id="activity"><div className="panel-heading"><div><div className="eyebrow">RECENT ACTIVITY</div><h2>Telemetry stream</h2></div><div className="activity-controls"><span className="activity-live"><i />{socketState === "online" ? "Streaming live" : "Reconnecting"}</span><input value={activitySearch} onChange={(event) => setActivitySearch(event.target.value)} placeholder="Filter this window" aria-label="Filter telemetry rows" /><button className="utility-button" onClick={() => setStreamPaused((value) => !value)}>{streamPaused ? "Resume" : "Pause"}</button><button className="utility-button primary" disabled={!path.length} onClick={exportTelemetry}>Export CSV</button></div></div><div className="table-scroll"><table><thead><tr><th>TIME</th><th>STATE OF CHARGE</th><th>PACK CURRENT</th><th>TEMPERATURE</th><th>SPEED</th><th>STATUS</th></tr></thead><tbody>{visibleSamples.slice(-8).reverse().map((sample, index) => { const needsReview = Number(sample.cell_temp_max) >= 48 || (Number(sample.speed_kmh) < 3 && Math.abs(Number(sample.pack_current)) > 45); return <tr key={`${sample.time}-${index}`}><td>{sample.time ? timeFmt.format(new Date(sample.time)) : "—"}</td><td>{fmt(sample.soc_reported, "%")}</td><td>{fmt(sample.pack_current, " A")}</td><td>{fmt(sample.cell_temp_max, " °C")}</td><td>{fmt(sample.speed_kmh, " km/h")}</td><td><span className={`row-status ${needsReview ? "needs-review" : ""}`}><i />{needsReview ? "Review" : "Recorded"}</span></td></tr>; })}</tbody></table>{!visibleSamples.length && <div className="empty-stream">{path.length ? "No telemetry rows match this filter." : "Telemetry samples will appear here when the vehicle connects."}</div>}</div></section>

          <section className="panel research-panel" id="research"><div className="panel-heading"><div><div className="eyebrow">PAPER-ALIGNED EVALUATION</div><h2>Research results</h2></div><button className="utility-button" disabled={!researchReport} onClick={() => { const link = document.createElement("a"); link.href = URL.createObjectURL(new Blob([JSON.stringify(researchReport, null, 2)], { type: "application/json" })); link.download = "aura-evaluation.json"; link.click(); URL.revokeObjectURL(link.href); }}>Export report</button></div>
            {researchReport ? <><div className="research-grid"><div><span>SoC MODEL MAE</span><strong>{fmt(researchModules.soc_range_cnn_bilstm?.soc?.mae, " pp")}</strong><small>RMSE {fmt(researchModules.soc_range_cnn_bilstm?.soc?.rmse)} pp</small></div><div><span>RANGE PROXY MAE</span><strong>{fmt(researchModules.soc_range_cnn_bilstm?.range_proxy?.mae, " km")}</strong><small>Derived label, not measured range</small></div><div><span>ANOMALY F1</span><strong>{fmt(researchModules.isolation_forest_vs_failure_label_proxy?.f1)}</strong><small>Failure label used as proxy</small></div><div><span>SHIELDED DQN MEAN COST</span><strong>{fmt(researchModules.smart_charging_dqn?.dqn_mean_cost, " USD")}</strong><small>Completion {fmt(Number(researchModules.smart_charging_dqn?.dqn_target_completion_rate) * 100, "%")} · raw DQN {fmt(Number(researchModules.smart_charging_dqn?.raw_dqn_target_completion_rate) * 100, "%")}</small></div><div><span>INFERENCE P95</span><strong>{fmt(researchReport.deployment?.latency_p95_ms, " ms")}</strong><small>Local CPU · machine-specific</small></div></div><details className="research-details"><summary>Full evaluation scorecard and limitations</summary><div className="research-table-wrap"><table className="research-table"><thead><tr><th>Module</th><th>Holdout metrics</th><th>Evaluation notes</th></tr></thead><tbody>{scorecard.map(([name, metrics, note]) => <tr key={name}><td>{name}</td><td>{metrics}</td><td>{note}</td></tr>)}</tbody></table></div><div className="research-resource-note">Inference: mean {fmt(researchReport.deployment?.latency_mean_ms)} ms, p50 {fmt(researchReport.deployment?.latency_p50_ms)} ms, p95 {fmt(researchReport.deployment?.latency_p95_ms)} ms · memory {fmt(researchReport.deployment?.process_memory_mb_after_benchmark)} MB · CPU sample {fmt(researchReport.deployment?.process_cpu_percent_during_benchmark)}% · training {fmt(researchReport.deployment?.training_total_seconds)} s.</div></details><div className="module-note">Chronological holdout metrics. Results are exploratory and inherit the Kaggle label, derived range target, and synthetic charging environment limitations. Export the JSON report for the full values.</div></> : <div className="empty-stream">Training evaluation is not available yet. Run <code>npm run setup</code> to train the models and produce the report.</div>}
          </section>
          <footer className="page-footer"><span>AURA EV Intelligence <b>·</b> Autonomous Unified Reasoning Architecture</span><span><Icon name="clock" size={13} /> {lastSeen ? `Telemetry received ${timeFmt.format(lastSeen)}` : "Waiting for first telemetry sample"}</span></footer>
        </div>
      </main>
    </div>
  );
}

export default App;
