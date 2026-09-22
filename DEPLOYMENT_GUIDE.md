# Azure Cloud Production Deployment Guide ($0.00 Cost Model)

This guide documents the complete production cloud deployment of **Cognitive EW SmartScan** on **Microsoft Azure** using the **GitHub Student Developer Pack (Azure for Students)**.

---

## 1. Cloud Architecture Overview

The system is deployed across an integrated, enterprise-grade architecture in Azure Region `indiasouthcentral`:

```
                                    +---------------------------------------------------+
                                    |         Azure Region: indiasouthcentral           |
                                    |                                                   |
  +--------------------------+      |  +---------------------------------------------+  |
  |  GitHub Container        |      |  |           smartscan-rg (Resource Group)     |  |
  |  Registry (GHCR)         |      |  |                                             |  |
  |  ghcr.io/promothesh-     |      |  |  +---------------------------------------+  |  |
  |  chatterjee/sih2026_try2 |====> |  |  |     Azure Kubernetes Service (AKS)    |  |  |
  |  (:latest, 3.24 GB)      |      |  |  |     Cluster: smartscan-aks            |  |  |
  +--------------------------+      |  |  |     Node: 1x Standard_B2s (Burstable) |  |  |
                                    |  |  |     Public IP: 172.198.227.59:80      |  |  |
                                    |  |  +---------------------------------------+  |  |
                                    |  |                         |                   |  |
                                    |  |                         v                   |  |
                                    |  |  +---------------------------------------+  |  |
                                    |  |  |      Storage: smartscanstore4301      |  |  |
                                    |  |  |      - Container: tsrd-dataset        |  |  |
                                    |  |  |      - Container: smartscan-models    |  |  |
                                    |  |  |      - Container: reports             |  |  |
                                    |  |  +---------------------------------------+  |  |
                                    |  +---------------------------------------------+  |
                                    +---------------------------------------------------+
```

---

## 2. Resource Inventory & Free Tier ($0.00) Accounting

Every component has been configured to guarantee zero out-of-pocket costs:

| Resource | Service / Name | Configuration / Tier | Region | Cost Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Resource Group** | `smartscan-rg` | Management container | `indiasouthcentral` | **$0.00** (Free) |
| **Container Registry** | GitHub Container Registry (GHCR) | `ghcr.io/promothesh-chatterjee/sih2026_try2:latest` | Global (CDN) | **$0.00** (Included in GitHub Student Pack) |
| **Blob Storage Account** | `smartscanstore4301` | `Standard_LRS` (5 GB Free Tier) | `indiasouthcentral` | **$0.00** (~180 MB used, < 4% of free quota) |
| **Blob Containers** | `tsrd-dataset`, `smartscan-models`, `reports` | Private Blob Storage | `indiasouthcentral` | **$0.00** |
| **Kubernetes Cluster** | `smartscan-aks` | 1 Node `Standard_B2s` (Linux 64-bit) | `indiasouthcentral` | Covered by Azure Student credits; **$0.00 when paused** |
| **Public LoadBalancer** | `smartscan-api-svc` | Standard Public IP (`172.198.227.59:80`) | `indiasouthcentral` | **$0.00** (Attached to cluster) |

> [!IMPORTANT]
> **Azure Student Policy Compliance**: All Azure resources are strictly deployed in region **`indiasouthcentral`** to comply with the Azure for Students policy (`sys.regionrestriction`).

---

## 3. Cluster Lifecycle Management (Freeze & Resume)

To ensure zero compute credits are consumed when not performing live tests or evaluations, use the following single-command Azure CLI operations:

### A. Resume / Start Cluster (Before Demonstrations)
To wake up the Kubernetes cluster and bring all pods and APIs online:
```powershell
az aks start --name smartscan-aks --resource-group smartscan-rg
```
* **Execution Time**: ~2 minutes.
* **Result**: The `Standard_B2s` VM node is reallocated, the container image is loaded from local cache, and the service resumes immediately on **`http://172.198.227.59`**.

### B. Pause / Stop Cluster (After Demonstrations)
To pause compute execution and freeze billing at **$0.00**:
```powershell
az aks stop --name smartscan-aks --resource-group smartscan-rg
```
* **Execution Time**: ~1.5 minutes.
* **Result**: Deallocates the compute instance cores. Storage and IP configurations are preserved, but all compute consumption stops.

### C. Check Live Power State
To verify whether the cluster is currently active or paused:
```powershell
az aks show --name smartscan-aks --resource-group smartscan-rg --query "{PowerState:powerState.code,ProvisioningState:provisioningState}"
```
* Output when paused:
  ```json
  {
    "PowerState": "Stopped",
    "ProvisioningState": "Succeeded"
  }
  ```
* Output when running:
  ```json
  {
    "PowerState": "Running",
    "ProvisioningState": "Succeeded"
  }
  ```

---

## 4. Live Endpoint Reference & Verification

When the cluster is active, the following endpoints are available at Public IP **`172.198.227.59:80`**:

### 1. Readiness Probe
* **Method**: `GET`
* **URL**: `http://172.198.227.59/ready`
* **Response**:
  ```json
  {
    "status": "ready",
    "model_loaded": true,
    "dataset_root": "/mnt/tsrd",
    "scenarios_count": 0,
    "ts": 1790107024.0569508
  }
  ```

### 2. Multi-Scheduler Benchmark Comparison API
* **Method**: `GET`
* **URL**: `http://172.198.227.59/api/benchmark`
* **Description**: Serves the complete 4-scheduler comparative analysis table across all 5,000 receiver dwell steps (evaluating **SmartScan DRQN MoE**, **Random**, **RoundRobin**, and **HighestOccupancy**).

### 3. Live Metrics WebSocket Stream
* **Protocol**: `WebSocket`
* **URL**: `ws://172.198.227.59/ws/metrics`
* **Description**: Live, bidirectional WebSocket channel streaming real-time operational radar telemetry (interception rate, sensitivity, false alarms, and Figures of Merit).

### 4. Neural Action Inference
* **Method**: `POST`
* **URL**: `http://172.198.227.59/predict_bands`
* **Payload**:
  ```json
  {
    "obs": [0.0, ..., 0.0],
    "policy_mode": "operational"
  }
  ```
* **Description**: Executes neural action arbitration using the frozen operational candidate checkpoint (`Gate-25k-R4.2-alpha020`) with calibrated inference latency (~54 ms).

---

## 5. Connecting the Frontend Dashboard

The frontend application (`frontend/`) is configured to interface directly with the live Azure AKS backend.

### Running Frontend Locally with Live Azure AKS Backend:
1. Open PowerShell in `frontend/`:
   ```powershell
   npm run preview
   ```
2. The dashboard runs at `http://localhost:4173/` (or `http://localhost:5173/`) and connects across the internet to the Azure AKS backend at `http://172.198.227.59` and `ws://172.198.227.59/ws/metrics`.
3. All CORS headers on the FastAPI backend are configured to authorize local development and preview origins.

---

## 6. Kubernetes Deployment Manifests

The Kubernetes configuration files are maintained in the `k8s/` directory:
* [`k8s/deployment.yaml`](file:///c:/Users/PromotheshChatterjee/Documents/GitHub/SIH2026_Try2/k8s/deployment.yaml): Contains the Deployment and Public LoadBalancer Service specs, with `/ready` liveness and readiness probes, resource limits (1 CPU, 2 GiB RAM), and volume mounts.
* `ghcr-secret`: Secret configured on the cluster containing GitHub PAT credentials to pull from GitHub Container Registry.
* `smartscan-secrets`: Secret configured with the Azure Storage Account connection string and API keys.
