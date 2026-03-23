# AWS GPU Setup

This repo is designed to run full experiments on an AWS Ubuntu 22.04 GPU instance with Docker and the NVIDIA Container Toolkit.

## Recommended Host

- Default: `g6e.4xlarge`
- Budget fallback: `g6.4xlarge`
- Escalation only if needed: `p5.4xlarge`

## Host Steps

1. Verify the NVIDIA driver on the host with `scripts/aws/check_driver.sh`.
2. Install Docker Engine with `scripts/aws/install_docker_ubuntu.sh`.
3. Install the NVIDIA Container Toolkit with `scripts/aws/install_nvidia_toolkit_ubuntu.sh`.
4. Configure Docker for the NVIDIA runtime with `scripts/aws/configure_docker_runtime.sh`.
5. Validate GPU passthrough with `scripts/aws/validate_cuda_container.sh`.

## Repo Layout On AWS

- Repo checkout: `/workspace/creditlab`
- Active runs/checkpoints: `/data/creditlab`
- Optional S3 sync target: configure `deployment.s3_sync_uri` and use `scripts/aws/sync_to_s3.sh`

## Containers

- CPU/dev image: `docker/Dockerfile.cpu`
- GPU image: `docker/Dockerfile.gpu`
- Single-machine composition: `docker/docker-compose.aws.yml`

## Validation

Run:

```bash
python3 -m creditlab.cli.main host-check --output /data/creditlab/host_report.json
docker compose -f docker/docker-compose.aws.yml config
```

