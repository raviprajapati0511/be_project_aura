# Research paper alignment

Compared against the attached paper, "Machine Learning for Unified Electric Vehicle Monitoring and Management: A Review of State Estimation, Smart Charging, Predictive Maintenance and Cybersecurity, with a Modular Framework Proposal" (AURA-EV research direction).
Requirements described in the paper are treated as project requirements; the
paper itself is a research proposal/review and does not provide a validated
production controller or a complete experimental dataset.

## Implemented modules

| Paper capability | Project implementation | Evidence / qualification |
| --- | --- | --- |
| Telemetry acquisition | Validated HTTP ingestion, socket broadcast, MySQL storage, and replay of Kaggle records | A physical OBD-II/CAN adapter is not included; simulator and replay are demonstration inputs. |
| Time-series management | Timestamped append-only telemetry rows with vehicle/time lookup and fleet latest-row queries | MySQL is used for the local project; not a specialized TSDB benchmark. |
| Kalman preprocessing | Per-feature scalar Kalman smoothing of standardized temporal features | Filtered sequence is fed to the sequence model; it is not sensor fusion from physical devices. |
| CNN + BiLSTM SoC/range estimation | Multi-head PyTorch model with chronological 80/20 holdout and SoC sensor-noise baseline | Source SoC is a label; range is a documented proxy from SoC, SoH, power, and speed, not measured range. |
| Anomaly detection | Isolation Forest; threshold and cross-domain fusion ablations | Kaggle failure label is only used as a proxy for anomaly ground truth, so it is not fault taxonomy or cyberattack detection. |
| Predictive maintenance | Random Forest failure classifier and RUL regressor | Uses published dataset labels; RUL unit is unspecified by the source. |
| Smart charging | Shielded DQN in a 15-minute Gymnasium environment plus Random Forest energy/duration forecasts trained on Kaggle EV Charging Patterns | The policy environment is synthetic; the separate session model uses a chronological Kaggle holdout. Neither controls a charger. |
| CAN intrusion detection | Random Forest on 100-frame traffic windows with rate, inter-arrival, CAN-ID, DLC, and payload features; HCRL injected-message flags are used only to label training windows; API accepts CAN frame batches | HCRL Kaggle mirror classes: benign, DoS, fuzzy, RPM spoofing, gear spoofing. Probabilities are uncalibrated; later windows from each class are held out within capture files, not an unseen vehicle or continuous future stream. |
| Cross-domain reasoning | Power-versus-motion residual proxy, telemetry anomaly, maintenance risk, and CAN IDS score fused into transparent hypotheses; attack pattern defers the charge advisory | Heuristic patterns only; residual thresholds are uncalibrated. The telemetry simulator does not synthesize CAN data, so CAN input must come from captured frames. |
| Dashboard and fleet decisions | Fleet table, live telemetry, charging panel, alert signals, security/data-quality panel, evaluation view, report export, CSV export | Displayed maps/replay IDs can be synthetic; they are not included in the Kaggle source. |
| Evaluation | Chronological holdout metrics; regression MAE/RMSE; precision/recall/F1/FPR; threshold/fusion baselines; DQN/rule comparison; training memory and API inference latency | See `services/api/models/evaluation.json` after model training. Metrics inherit proxy-label and synthetic-environment limitations. |

## Not implementable from the supplied data alone

- Physical OBD-II/CAN acquisition requires an adapter, vehicle protocol mapping,
  and hardware access. The project exposes an ingestion contract and simulator
  instead.
- Real-time vehicle CAN acquisition still requires an adapter and vehicle
  protocol access. The project trains a windowed classifier on HCRL captures and
  exposes a batch inference endpoint; it does not connect directly to a vehicle
  bus or validate cross-vehicle generalization.
- Measured range validation requires observed vehicle range or energy-to-empty
  outcomes. The current range target is explicitly a derived proxy.
- Operational smart charging requires a real station/utility integration,
  measured tariff and grid constraints, and safety validation. The DQN remains
  an offline research demonstration.
- Edge-device latency/CPU comparisons, robustness to real sensor faults, and
  external vehicle validation require target hardware and field data. The report
  records local CPU training resource use and request inference latency only.

## Reproduce training

On Windows, `npm run setup` installs dependencies, downloads Kaggle EV charging and HCRL CAN files, and trains all local modules. The downloaded raw CAN captures total roughly 1 GB; Kaggle account/API access may be required.
To download the auxiliary sources later, run `npm run data`. To retrain after setup, run:

```powershell
.\.venv\Scripts\python services\api\scripts\train_models.py
```

Training reports include dataset provenance, chronological train/test splits, per-task metrics, and known generalization limits. Training writes model weights and `evaluation.json` under
`services/api/models/`. The Kaggle source archive and its CC BY-NC-SA 4.0
attribution are documented in the root README.
