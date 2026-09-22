<#
.SYNOPSIS
    SmartScan EW — Azure Infrastructure Deployment Script
.DESCRIPTION
    Automated deployment script tailored for Azure for Students subscription.
    Provisions Resource Group, ACR, Storage Account (TSRD dataset & models),
    builds the Docker image in Azure via ACR Cloud Build (no local Docker daemon required),
    provisions AKS with Blob CSI driver, deploys the backend API, and deploys
    the React frontend to Azure Static Web Apps.
.PARAMETER ResourceGroup
    Azure Resource Group name (default: smartscan-rg)
.PARAMETER Location
    Azure Region (default: centralindia or eastus)
.PARAMETER AcrName
    Unique Azure Container Registry name (alphanumeric only, 5-50 chars)
.PARAMETER StorageAccountName
    Unique Storage Account name (alphanumeric only, 3-24 chars)
.PARAMETER AksName
    AKS cluster name (default: smartscan-aks)
.PARAMETER VmSize
    VM size for AKS nodes (default: Standard_B2s for student quota safety)
.PARAMETER NodeCount
    Number of worker nodes in AKS (default: 2)
.PARAMETER TsrdLocalPath
    Local path to TSRD dataset on host machine (default: D:/TSRD)
.PARAMETER ApiKey
    SmartScan API Key for authentication (default: smartscan-sih2026-demo-key)
#>

[CmdletBinding()]
param(
    [string]$ResourceGroup = "smartscan-rg",
    [string]$Location = "centralindia",
    [string]$AcrName = "",
    [string]$StorageAccountName = "",
    [string]$AksName = "smartscan-aks",
    [string]$VmSize = "Standard_B2s",
    [int]$NodeCount = 2,
    [string]$TsrdLocalPath = "D:/TSRD",
    [string]$ApiKey = "smartscan-sih2026-demo-key",
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
# 0. Check Azure CLI & Login Status
# -----------------------------------------------------------------------------
Write-Step "0. Checking Azure CLI & Authentication"

# Refresh environment PATH in current PowerShell session
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
    Write-Error "Azure CLI ('az') is not installed or not found in PATH. Please install Azure CLI or restart terminal."
    exit 1
}

Write-Info "Verifying active Azure login..."
$accountJson = az account show 2>$null
if (-not $accountJson) {
    Write-Host "`nYou are not logged into Azure CLI. Running 'az login'..." -ForegroundColor Yellow
    az login --output table
    $accountJson = az account show
}

$account = $accountJson | ConvertFrom-Json
Write-Success "Logged into Azure Subscription: '$($account.name)' (ID: $($account.id))"

# -----------------------------------------------------------------------------
# 1. Generate Unique Resource Names if Not Provided
# -----------------------------------------------------------------------------
Write-Step "1. Configuring Resource Names"

if ([string]::IsNullOrWhiteSpace($AcrName)) {
    $randomSuffix = Get-Random -Minimum 1000 -Maximum 9999
    $AcrName = "smartscanacr$randomSuffix"
}
if ([string]::IsNullOrWhiteSpace($StorageAccountName)) {
    $randomSuffix = Get-Random -Minimum 1000 -Maximum 9999
    $StorageAccountName = "smartscanstore$randomSuffix"
}

Write-Info "Resource Group:        $ResourceGroup"
Write-Info "Location:              $Location"
Write-Info "ACR Name:              $AcrName"
Write-Info "Storage Account:       $StorageAccountName"
Write-Info "AKS Cluster Name:      $AksName"
Write-Info "AKS Node VM Size:      $VmSize (Node count: $NodeCount)"
Write-Info "TSRD Local Source:     $TsrdLocalPath"

# -----------------------------------------------------------------------------
# 2. Register Required Azure Resource Providers
# -----------------------------------------------------------------------------
Write-Step "2. Registering Azure Resource Providers"

$providers = @(
    "Microsoft.ContainerRegistry",
    "Microsoft.ContainerService",
    "Microsoft.Storage",
    "Microsoft.Web"
)

foreach ($p in $providers) {
    Write-Info "Ensuring provider registration: $p"
    az provider register --namespace $p --output none 2>$null
}
Write-Success "Resource providers registered."

# -----------------------------------------------------------------------------
# 3. Create Resource Group
# -----------------------------------------------------------------------------
Write-Step "3. Creating Resource Group '$ResourceGroup' in '$Location'"
az group create --name $ResourceGroup --location $Location --output table
Write-Success "Resource Group created/verified."

# -----------------------------------------------------------------------------
# 4. Create Azure Container Registry (ACR)
# -----------------------------------------------------------------------------
Write-Step "4. Creating Azure Container Registry '$AcrName'"
az acr create `
    --resource-group $ResourceGroup `
    --name $AcrName `
    --sku Basic `
    --admin-enabled true `
    --output table
Write-Success "ACR '$AcrName' ready."

# -----------------------------------------------------------------------------
# 5. Create Azure Storage Account & Containers
# -----------------------------------------------------------------------------
Write-Step "5. Creating Storage Account '$StorageAccountName' & Containers"

az storage account create `
    --resource-group $ResourceGroup `
    --name $StorageAccountName `
    --location $Location `
    --sku Standard_LRS `
    --kind StorageV2 `
    --allow-blob-public-access false `
    --output table

Write-Info "Retrieving Storage Connection String..."
$connString = (az storage account show-connection-string `
    --resource-group $ResourceGroup `
    --name $StorageAccountName `
    --query connectionString `
    --output tsv)

$containers = @("tsrd-dataset", "smartscan-models", "reports")
foreach ($c in $containers) {
    Write-Info "Creating container '$c'..."
    az storage container create `
        --name $c `
        --connection-string $connString `
        --output none
}
Write-Success "Storage account and containers ready."

# -----------------------------------------------------------------------------
# 6. Upload Dataset and Checkpoints
# -----------------------------------------------------------------------------
Write-Step "6. Uploading Assets to Azure Blob Storage"

if (-not $SkipDatasetUpload) {
    if (Test-Path $TsrdLocalPath) {
        Write-Info "Uploading TSRD dataset from $TsrdLocalPath to container 'tsrd-dataset'..."
        az storage blob upload-batch `
            --destination "tsrd-dataset" `
            --source $TsrdLocalPath `
            --connection-string $connString `
            --output table
        Write-Success "TSRD dataset uploaded."
    } else {
        Write-Host "[WARNING] Local TSRD path '$TsrdLocalPath' not found. Skipping dataset batch upload." -ForegroundColor Yellow
        Write-Host "          You can upload later with 'az storage blob upload-batch --destination tsrd-dataset --source <PATH>'." -ForegroundColor Yellow
    }

    # Upload best model checkpoint if available
    $checkpointPath = "experiments/checkpoints/best_model.zip"
    if (Test-Path $checkpointPath) {
        Write-Info "Uploading frozen model checkpoint '$checkpointPath' to 'smartscan-models'..."
        az storage blob upload `
            --container-name "smartscan-models" `
            --file $checkpointPath `
            --name "best_model.zip" `
            --connection-string $connString `
            --output table
        Write-Success "Frozen model checkpoint uploaded."
    }
} else {
    Write-Info "Skipping dataset upload per --SkipDatasetUpload flag."
}

# -----------------------------------------------------------------------------
# 7. Build and Push Container Image to ACR (Cloud Build)
# -----------------------------------------------------------------------------
Write-Step "7. Building & Pushing Docker Image using Azure ACR Cloud Build"
Write-Info "Building 'smartscan-api:latest' directly in Azure compute (no local Docker engine required)..."

az acr build `
    --registry $AcrName `
    --image "smartscan-api:latest" `
    --file Dockerfile `
    .

Write-Success "Docker image built and stored in $AcrName.azurecr.io/smartscan-api:latest"

# -----------------------------------------------------------------------------
# 8. Create AKS Cluster (Azure for Students Safe)
# -----------------------------------------------------------------------------
if (-not $SkipAks) {
    Write-Step "8. Provisioning Azure Kubernetes Service (AKS) Cluster '$AksName'"
    Write-Info "Cluster node configuration: $NodeCount nodes of size $VmSize (total: $($NodeCount * 2) vCPUs, fits student quota)"

    az aks create `
        --resource-group $ResourceGroup `
        --name $AksName `
        --node-count $NodeCount `
        --node-vm-size $VmSize `
        --attach-acr $AcrName `
        --enable-blob-driver `
        --generate-ssh-keys `
        --output table

    Write-Success "AKS Cluster '$AksName' provisioned."

    # Get credentials for kubectl
    Write-Info "Configuring kubectl context for '$AksName'..."
    az aks get-credentials --resource-group $ResourceGroup --name $AksName --overwrite-existing

    # -----------------------------------------------------------------------------
    # 9. Deploy Backend to AKS
    # -----------------------------------------------------------------------------
    Write-Step "9. Deploying SmartScan API to AKS"

    # Create / update Kubernetes Secret
    Write-Info "Applying smartscan-secrets Kubernetes Secret..."
    kubectl delete secret smartscan-secrets --ignore-not-found=true
    kubectl create secret generic smartscan-secrets `
        --from-literal=api-key="$ApiKey" `
        --from-literal=azure-storage-connection-string="$connString"

    # Apply PersistentVolumeClaim for TSRD Blob
    if (Test-Path "k8s/blob-storage-pvc.yaml") {
        Write-Info "Applying Blob Storage PVC..."
        kubectl apply -f k8s/blob-storage-pvc.yaml
    }

    # Render and apply deployment.yaml with ACR name substituted
    Write-Info "Deploying smartscan-api with image '$AcrName.azurecr.io/smartscan-api:latest'..."
    $deploymentManifest = Get-Content "k8s/deployment.yaml" -Raw
    $renderedManifest = $deploymentManifest -replace '\$\{ACR_NAME\}', $AcrName
    $renderedManifest | kubectl apply -f -

    Write-Info "Waiting for smartscan-api deployment rollout..."
    kubectl rollout status deployment/smartscan-api --timeout=300s

    Write-Info "Fetching Public LoadBalancer IP (this may take 1-2 minutes)..."
    $backendIp = ""
    for ($i = 0; $i -lt 30; $i++) {
        $backendIp = (kubectl get svc smartscan-api-svc -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>$null)
        if (-not [string]::IsNullOrWhiteSpace($backendIp)) {
            break
        }
        Start-Sleep -Seconds 5
    }

    if ($backendIp) {
        Write-Success "SmartScan API Public IP: http://$backendIp"
        Write-Success "API Health Endpoint:     http://$backendIp/health"
        Write-Success "API Docs:                http://$backendIp/docs"
        Write-Success "WebSocket Metrics:       ws://$backendIp/ws/metrics"
    } else {
        Write-Host "[NOTICE] LoadBalancer IP is still provisioning. Run 'kubectl get svc smartscan-api-svc' in a few minutes." -ForegroundColor Yellow
    }
}

# -----------------------------------------------------------------------------
# 10. Frontend Build & Azure Static Web Apps Deployment
# -----------------------------------------------------------------------------
if (-not $SkipFrontend) {
    Write-Step "10. Building and Deploying Frontend to Azure Static Web Apps"

    $frontendDir = Join-Path $PSScriptRoot "..\frontend"
    if (Test-Path $frontendDir) {
        Push-Location $frontendDir

        # If backend IP is known, set as build env var
        if ($backendIp) {
            $env:VITE_API_URL = "http://$backendIp"
            $env:VITE_WS_URL = "ws://$backendIp/ws/metrics"
        }

        Write-Info "Building production React frontend with Vite..."
        npm run build

        Write-Info "Creating/Deploying Azure Static Web App '$StaticWebAppName'..."
        # If az staticwebapp extension is not installed, install it
        az extension add --name staticwebapps --allow-preview $true --output none 2>$null

        az staticwebapp create `
            --name $StaticWebAppName `
            --resource-group $ResourceGroup `
            --location $Location `
            --source . `
            --output table

        Pop-Location
        Write-Success "Frontend deployed to Azure Static Web Apps!"
    } else {
        Write-Host "[WARNING] Frontend directory '$frontendDir' not found. Skipping frontend deployment." -ForegroundColor Yellow
    }
}

Write-Step "DEPLOYMENT COMPLETE"
Write-Host "SmartScan EW Multi-Node Defense Suite is fully configured for Azure." -ForegroundColor Green
if ($backendIp) {
    Write-Host "Access the Backend API at: http://$backendIp" -ForegroundColor Cyan
    Write-Host "API Key: $ApiKey" -ForegroundColor Cyan
}
