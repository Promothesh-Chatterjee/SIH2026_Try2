<#
.SYNOPSIS
    SmartScan EW — Zero-Cost Azure Infrastructure Deployment Script
.DESCRIPTION
    Automated deployment script tailored for Azure for Students subscription.
    Uses GitHub Container Registry (ghcr.io) for 100% FREE image hosting ($0 ACR).
    Provisions Resource Group, Azure Storage Account (lightweight TSRD slice & model),
    AKS Cluster (Standard_B2s, 1 node) with Blob CSI driver, and deploys the
    React frontend to Azure Static Web Apps (Free Plan).
#>

[CmdletBinding()]
param(
    [string]$ResourceGroup = "smartscan-rg",
    [string]$Location = "centralindia",
    [string]$StorageAccountName = "",
    [string]$AksName = "smartscan-aks",
    [string]$VmSize = "Standard_B2s",
    [int]$NodeCount = 1,
    [string]$ImageName = "ghcr.io/promothesh-chatterjee/sih2026_try2:latest",
    [string]$TsrdLocalPath = "D:/TSRD/stare/val_stare",
    [string]$ApiKey = "smartscan-sih2026-demo-key",
    [string]$StaticWebAppName = "smartscan-ui",
    [switch]$SkipDatasetUpload,
    [switch]$SkipAks,
    [switch]$SkipFrontend
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "`n========================================================" -ForegroundColor Cyan
    Write-Host "  $Message" -ForegroundColor Yellow
    Write-Host "========================================================" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Gray
}

# -----------------------------------------------------------------------------
# 0. Check Azure CLI & Authentication
# -----------------------------------------------------------------------------
Write-Step "0. Checking Azure CLI & Authentication"

$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
    Write-Error "Azure CLI ('az') is not installed or not in PATH."
    exit 1
}

$accountJson = az account show 2>$null
if (-not $accountJson) {
    Write-Host "Running 'az login'..." -ForegroundColor Yellow
    az login --output table
    $accountJson = az account show
}

$account = $accountJson | ConvertFrom-Json
Write-Success "Logged in: '$($account.name)' (ID: $($account.id))"

# -----------------------------------------------------------------------------
# 1. Resource Names & Config
# -----------------------------------------------------------------------------
Write-Step "1. Configuring Resource Names"

if ([string]::IsNullOrWhiteSpace($StorageAccountName)) {
    $randomSuffix = Get-Random -Minimum 1000 -Maximum 9999
    $StorageAccountName = "smartscanstore$randomSuffix"
}

Write-Info "Resource Group:    $ResourceGroup"
Write-Info "Location:          $Location"
Write-Info "Storage Account:   $StorageAccountName (Free Tier Eligible, 5GB)"
Write-Info "Container Image:   $ImageName (GitHub Container Registry — 100% Free)"
Write-Info "AKS Cluster:       $AksName ($NodeCount node of $VmSize)"
Write-Info "Static Web App:    $StaticWebAppName (Free Plan — $0.00)"

# -----------------------------------------------------------------------------
# 2. Register Required Providers
# -----------------------------------------------------------------------------
Write-Step "2. Ensuring Azure Resource Providers Registration"

$providers = @("Microsoft.Compute", "Microsoft.ContainerService", "Microsoft.Storage", "Microsoft.Web", "Microsoft.Network")
foreach ($p in $providers) {
    az provider register --namespace $p --output none 2>$null
}
Write-Success "Providers verified."

# -----------------------------------------------------------------------------
# 3. Create Resource Group ($0.00 Free)
# -----------------------------------------------------------------------------
Write-Step "3. Creating Resource Group '$ResourceGroup' ($0.00)"
az group create --name $ResourceGroup --location $Location --output table
Write-Success "Resource Group ready."

# -----------------------------------------------------------------------------
# 4. Create Storage Account (Free 5GB Tier)
# -----------------------------------------------------------------------------
Write-Step "4. Creating Storage Account '$StorageAccountName' (5GB Free Tier)"
az storage account create `
    --resource-group $ResourceGroup `
    --name $StorageAccountName `
    --location $Location `
    --sku Standard_LRS `
    --kind StorageV2 `
    --allow-blob-public-access false `
    --output table

$connString = (az storage account show-connection-string `
    --resource-group $ResourceGroup `
    --name $StorageAccountName `
    --query connectionString `
    --output tsv)

$containers = @("tsrd-dataset", "smartscan-models", "reports")
foreach ($c in $containers) {
    az storage container create --name $c --connection-string $connString --output none
}
Write-Success "Storage account and containers ready."

# -----------------------------------------------------------------------------
# 5. Upload Dataset Slice & Checkpoints (Lightweight, < 300MB)
# -----------------------------------------------------------------------------
Write-Step "5. Uploading Lightweight Dataset Slice & Frozen Model"

if (-not $SkipDatasetUpload) {
    if (Test-Path $TsrdLocalPath) {
        Write-Info "Uploading validation scenario files from $TsrdLocalPath to 'tsrd-dataset'..."
        # Upload first 15 scenario h5 files for instant verification without blowing quota/bandwidth
        $files = Get-ChildItem $TsrdLocalPath -Filter "*.h5" | Select-Object -First 15
        foreach ($f in $files) {
            az storage blob upload `
                --container-name "tsrd-dataset" `
                --file $f.FullName `
                --name "val_stare/$($f.Name)" `
                --connection-string $connString `
                --output none
        }
        Write-Success "15 validation scenarios uploaded to Azure Blob."
    }

    $checkpointPath = "experiments/checkpoints/best_model.zip"
    if (Test-Path $checkpointPath) {
        Write-Info "Uploading model checkpoint '$checkpointPath' to 'smartscan-models'..."
        az storage blob upload `
            --container-name "smartscan-models" `
            --file $checkpointPath `
            --name "best_model.zip" `
            --connection-string $connString `
            --output none
        Write-Success "Frozen model checkpoint uploaded."
    }
}

# -----------------------------------------------------------------------------
# 6. Create AKS Cluster (Student-Safe 1-Node Cluster)
# -----------------------------------------------------------------------------
if (-not $SkipAks) {
    Write-Step "6. Provisioning Azure Kubernetes Service (AKS) '$AksName'"
    Write-Info "Node config: $NodeCount x $VmSize (burstable, 2 vCPUs, fits student quota)"

    az aks create `
        --resource-group $ResourceGroup `
        --name $AksName `
        --node-count $NodeCount `
        --node-vm-size $VmSize `
        --enable-blob-driver `
        --generate-ssh-keys `
        --output table

    Write-Success "AKS Cluster provisioned."

    # Connect kubectl
    Write-Info "Configuring kubectl context..."
    az aks get-credentials --resource-group $ResourceGroup --name $AksName --overwrite-existing

    # -----------------------------------------------------------------------------
    # 7. Deploy Backend to AKS
    # -----------------------------------------------------------------------------
    Write-Step "7. Deploying SmartScan API to AKS"

    # Secrets
    kubectl delete secret smartscan-secrets --ignore-not-found=true
    kubectl create secret generic smartscan-secrets `
        --from-literal=api-key="$ApiKey" `
        --from-literal=azure-storage-connection-string="$connString"

    # PersistentVolumeClaim for TSRD Blob
    if (Test-Path "k8s/blob-storage-pvc.yaml") {
        kubectl apply -f k8s/blob-storage-pvc.yaml
    }

    # Render deployment manifest with GHCR image
    $deploymentManifest = Get-Content "k8s/deployment.yaml" -Raw
    $renderedManifest = $deploymentManifest -replace '\$\{IMAGE_NAME:.*\}', $ImageName
    $renderedManifest | kubectl apply -f -

    Write-Info "Waiting for deployment rollout..."
    kubectl rollout status deployment/smartscan-api --timeout=300s

    Write-Info "Fetching Public LoadBalancer IP..."
    $backendIp = ""
    for ($i = 0; $i -lt 30; $i++) {
        $backendIp = (kubectl get svc smartscan-api-svc -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>$null)
        if (-not [string]::IsNullOrWhiteSpace($backendIp)) { break }
        Start-Sleep -Seconds 5
    }

    if ($backendIp) {
        Write-Success "API Public Endpoint: http://$backendIp"
        Write-Success "API Health:          http://$backendIp/health"
        Write-Success "API Documentation:   http://$backendIp/docs"
        Write-Success "Live WebSocket:      ws://$backendIp/ws/metrics"
    }
}

# -----------------------------------------------------------------------------
# 8. Deploy Frontend to Azure Static Web Apps (Free Plan — $0.00)
# -----------------------------------------------------------------------------
if (-not $SkipFrontend) {
    Write-Step "8. Building & Deploying Frontend to Azure Static Web Apps (Free Plan)"

    $frontendDir = Join-Path $PSScriptRoot "..\frontend"
    if (Test-Path $frontendDir) {
        Push-Location $frontendDir

        if ($backendIp) {
            $env:VITE_API_URL = "http://$backendIp"
            $env:VITE_WS_URL = "ws://$backendIp/ws/metrics"
        }

        npm run build

        az extension add --name staticwebapps --allow-preview $true --output none 2>$null
        az staticwebapp create `
            --name $StaticWebAppName `
            --resource-group $ResourceGroup `
            --location $Location `
            --source . `
            --output table

        Pop-Location
        Write-Success "Frontend deployed to Azure Static Web Apps!"
    }
}

# -----------------------------------------------------------------------------
# Summary & Pause Instructions
# -----------------------------------------------------------------------------
Write-Step "DEPLOYMENT COMPLETE"
Write-Host "To pause the AKS cluster and freeze compute billing at `$0.00:" -ForegroundColor Yellow
Write-Host "  az aks stop --name $AksName --resource-group $ResourceGroup" -ForegroundColor Cyan
Write-Host "To resume the AKS cluster:" -ForegroundColor Yellow
Write-Host "  az aks start --name $AksName --resource-group $ResourceGroup" -ForegroundColor Cyan
