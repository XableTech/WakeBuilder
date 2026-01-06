# Deployment

Options for deploying and running WakeBuilder.

---

## Overview

WakeBuilder can be deployed in several ways:

| Method | Best For |
|--------|----------|
| **Local Development** | Personal use, testing, development |
| **Docker** | Consistent environment, easy deployment |
| **Production** | Server deployment, team use |

---

## In This Section

<div class="grid cards" markdown>

- :material-laptop:{ .lg .middle } **Local Development**

    ---

    Run WakeBuilder on your local machine for development and personal use.

    [:octicons-arrow-right-24: Local Setup](local-development.md)

- :material-docker:{ .lg .middle } **Docker Deployment**

    ---

    Deploy WakeBuilder in Docker containers for consistent, portable environments.

    [:octicons-arrow-right-24: Docker Guide](docker.md)

- :material-server:{ .lg .middle } **Production**

    ---

    Considerations for production deployment including security and scaling.

    [:octicons-arrow-right-24: Production Guide](production.md)

</div>

---

## Quick Comparison

| Feature | Local | Docker | Production |
|---------|-------|--------|------------|
| Setup time | 5-10 min | 10-15 min | Varies |
| GPU support | ✓ | ✓ (NVIDIA runtime) | ✓ |
| Isolation | None | Container | Full |
| Portability | Low | High | High |
| Recommended for | Development | Testing, sharing | Team use |

---

## Hardware Requirements

All deployment methods require:

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| **RAM** | 8GB free | 16GB+ free |
| **GPU VRAM** | 6GB | 8GB+ |
| **Storage** | 10GB | 20GB+ |
| **CPU** | 4 cores | 8+ cores |
