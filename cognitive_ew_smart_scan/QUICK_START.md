## 🚀 Quick Start — Cognitive EW SmartScan Dashboard

### 30-Second Setup

**Terminal 1 — Backend:**
```bash
cd cognitive_ew_smart_scan
uvicorn src.deployment.api:app --host 0.0.0.0 --port 8080 --reload
```

**Terminal 2 — Primary Frontend (`SMARTSCAN_EW_FRONTEND_APP`):**
```bash
cd SMARTSCAN_EW_FRONTEND_APP
npm run dev
```

---

### What to Expect

✅ **WebSocket: ONLINE**  
✅ Four navigation tabs: **OVERVIEW | SPECTRUM | METRICS | PDWS**  
✅ Real-time update stream from backend API

---

### Test the Connection

```bash
# Verify backend health
curl http://localhost:8080/health

# Test WebSocket (optional, requires wscat)
npm install -g wscat
wscat -c ws://localhost:8080/ws/state
```

---

### Files Structure

```
SIH2026_Try2/
├── cognitive_ew_smart_scan/          # Primary Backend & ML Engine
└── SMARTSCAN_EW_FRONTEND_APP/        # Primary Working Frontend App
```

---

### Troubleshooting at a Glance

| Issue | Fix |
|-------|-----|
| WebSocket shows OFFLINE | Verify backend running on port 8080: `curl http://localhost:8080/health` |
| No data in panels | Verify TSRD chunk stream or simulation is active |
| Port already in use | Change port: `uvicorn ... --port 8081` and update frontend wsUrl |

