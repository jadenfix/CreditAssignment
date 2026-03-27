# GCP GPU Setup

This repo supports running full experiments on a GCP Ubuntu 22.04 GPU instance with Docker and the NVIDIA Container Toolkit. This document covers provisioning, bootstrapping, and validation.

## Recommended Host

| Tier | Machine type | GPU | On-demand | Spot |
|---|---|---|---|---|
| Default | `g2-standard-8` | 1× NVIDIA L4 (24 GB) | ~$0.70/hr | ~$0.25/hr |
| Escalation | `g2-standard-48` | 4× NVIDIA L4 | ~$2.80/hr | ~$1.00/hr |

Zone: `us-central1-a` (good L4 availability; adjust as needed).

## Prerequisites

Install and authenticate the gcloud CLI locally:

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable compute.googleapis.com storage.googleapis.com
```

## Provision the VM

### On-demand

```bash
gcloud compute instances create creditlab-gpu \
  --project=YOUR_PROJECT_ID \
  --zone=us-central1-a \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --scopes=https://www.googleapis.com/auth/devstorage.read_write,https://www.googleapis.com/auth/logging.write
```

### Spot (add these flags to the on-demand command)

```bash
  --provisioning-model=SPOT \
  --instance-termination-action=STOP
```

Note: `--maintenance-policy=TERMINATE` is required for GPU instances (live migration is not supported).

## Create a GCS Bucket

```bash
gcloud storage buckets create gs://YOUR_BUCKET_creditlab \
  --project=YOUR_PROJECT_ID \
  --location=us-central1 \
  --uniform-bucket-level-access
```

## Copy the Repo to the VM

```bash
gcloud compute scp --recurse /path/to/creditlab \
  creditlab-gpu:/workspace/creditlab --zone=us-central1-a
```

Or clone from your git remote after SSH-ing in:

```bash
gcloud compute ssh creditlab-gpu --zone=us-central1-a
git clone <your-repo-url> /workspace/creditlab
```

## Host Bootstrap

Run all steps in one shot:

```bash
bash scripts/gcp/bootstrap_ubuntu22_gpu_gcp.sh
```

Or step by step:

1. `scripts/gcp/check_driver.sh` — verify the NVIDIA driver (`nvidia-smi`)
2. `scripts/gcp/install_docker_ubuntu.sh` — install Docker Engine
3. `scripts/gcp/install_nvidia_toolkit_ubuntu.sh` — install NVIDIA Container Toolkit
4. `scripts/gcp/configure_docker_runtime.sh` — set NVIDIA as default Docker runtime
5. `scripts/gcp/validate_cuda_container.sh` — test GPU passthrough via Docker

## Repo Layout on the VM

| Path | Purpose |
|---|---|
| `/workspace/creditlab` | Repo checkout |
| `/data/creditlab` | Active runs, SQLite DB, artifacts |
| `gs://YOUR_BUCKET_creditlab/runs` | Durable artifact backup (GCS) |

## Containers

| Image | Dockerfile | Purpose |
|---|---|---|
| `creditlab-gpu:latest` | `docker/Dockerfile.gpu` | Sweep worker, vllm-server |
| `creditlab-cpu:latest` | `docker/Dockerfile.cpu` | Exporter, analysis |

Compose stack: `docker/docker-compose.gcp.yml`

## Validation

After bootstrapping, confirm everything is wired up:

```bash
# GPU driver inside a container
docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi

# Config parses cleanly
creditlab validate-config --config configs/diagnostic_paper_gcp.yaml

# Full preflight (docker, nvidia-smi, nvidia-ctk, cuda probe)
creditlab host-check --output /data/creditlab/host_report.json

# Compose resolves with no errors (dry-run, no containers started)
docker compose -f docker/docker-compose.gcp.yml config
```

## Run the Sweep

1. Edit `configs/diagnostic_paper_gcp.yaml` and set:
   - `deployment.gcs_sync_uri: gs://YOUR_BUCKET_creditlab/runs`
   - `sweep.safety.allow_gcp_execution: true`

2. Launch:

```bash
creditlab sweep --config configs/diagnostic_paper_gcp.yaml
```

## Sync Artifacts to GCS

```bash
bash scripts/gcp/sync_to_gcs.sh /data/creditlab gs://YOUR_BUCKET_creditlab/runs
```

## Budget Reference

| Mode | Rate | Hours / $300 |
|---|---|---|
| On-demand `g2-standard-8` + L4 | ~$0.70/hr | ~428 hrs |
| Spot `g2-standard-8` + L4 | ~$0.25/hr | ~1200 hrs |

A full 3-seed diagnostic sweep (24 runs, ~30 min wall-clock) costs ~$0.35 on-demand or ~$0.12 spot. Validate on on-demand, then switch to spot for production runs.
