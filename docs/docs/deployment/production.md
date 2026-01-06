# Production Considerations

Guidelines for deploying WakeBuilder in production environments.

---

## ⚠️ Important Notes

WakeBuilder is currently in **beta** status. For production use:

- Thoroughly test with your use cases
- Implement proper monitoring
- Plan for updates and maintenance
- Consider security implications

---

## Security

### Network Security

**Restrict Access:**

```python
# In main.py, configure CORS properly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-domain.com"],  # Not "*"
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)
```

**Use HTTPS:**

Deploy behind a reverse proxy with TLS:

```nginx
# nginx.conf
server {
    listen 443 ssl;
    server_name wakebuilder.your-domain.com;
    
    ssl_certificate /etc/letsencrypt/live/your-domain/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain/privkey.pem;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

### Authentication

WakeBuilder does not include authentication. Implement it at the proxy level or extend the application:

```python
# Example: Simple API key auth
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.environ.get("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API key")
```

### File Upload Limits

Configure maximum upload sizes:

```python
# In FastAPI
app = FastAPI(
    max_request_size=10 * 1024 * 1024  # 10MB
)
```

---

## Performance

### Resource Requirements

| Users | RAM | CPU | GPU | Storage |
|-------|-----|-----|-----|---------|
| 1-2 | 16GB | 4 cores | Optional | 50GB |
| 5-10 | 32GB | 8 cores | Recommended | 100GB |
| 10+ | 64GB+ | 16+ cores | Required | 200GB+ |

### Optimizations

**Enable Production Mode:**

```bash
uvicorn src.wakebuilder.backend.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --loop uvloop \
    --http httptools
```

**Use Process Manager:**

```bash
# Using gunicorn with uvicorn workers
gunicorn src.wakebuilder.backend.main:app \
    -w 4 \
    -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000
```

### Caching

Pre-build the negative cache before deploying:

```bash
python scripts/build_negative_cache.py
```

---

## Scaling

### Horizontal Scaling

Training jobs are stateful and memory-intensive. For multiple users:

1. Use a job queue (Redis, RabbitMQ)
2. Run worker processes separately
3. Share model storage (NFS, S3)

### Vertical Scaling

Increase resources for single-instance deployment:

| Upgrade | Benefit |
|---------|---------|
| More RAM | More concurrent training |
| Better GPU | Faster training |
| Faster storage | Quicker data loading |

---

## Monitoring

### Health Endpoint

```bash
# Check health
curl http://localhost:8000/health

# Response
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2026-01-06T15:00:00Z"
}
```

### Metrics to Monitor

| Metric | Alert Threshold |
|--------|-----------------|
| Response time | > 2s average |
| Error rate | > 1% |
| Memory usage | > 80% |
| GPU memory | > 90% |
| Disk space | < 10GB free |

### Logging

Configure structured logging:

```python
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
        })
```

---

## Backup and Recovery

### What to Backup

| Directory | Priority | Contents |
|-----------|----------|----------|
| `models/custom/` | High | User-trained models |
| `data/cache/` | Medium | Preprocessed data (can rebuild) |
| `data/negative/` | Low | UNAC dataset (can redownload) |
| `tts_voices/` | Low | TTS models (can redownload) |

### Backup Script

```bash
#!/bin/bash
BACKUP_DIR="/backups/wakebuilder/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Backup models
tar -czf $BACKUP_DIR/models.tar.gz models/custom/

# Backup configuration
cp .env $BACKUP_DIR/
```

### Recovery

```bash
# Restore models
tar -xzf models.tar.gz -C /app/

# Rebuild cache
python scripts/build_negative_cache.py
```

---

## Maintenance

### Updates

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -e . --upgrade

# Restart service
systemctl restart wakebuilder
```

### Cleanup

```bash
# Remove old temporary files
python clean.py --temp

# Clear old models (interactive)
python clean.py --models

# Full cleanup
python clean.py --all
```

---

## Systemd Service

Create a systemd service for automatic startup:

```ini
# /etc/systemd/system/wakebuilder.service
[Unit]
Description=WakeBuilder Wake Word Training Platform
After=network.target

[Service]
Type=simple
User=wakebuilder
WorkingDirectory=/opt/wakebuilder
ExecStart=/opt/wakebuilder/.venv/bin/uvicorn src.wakebuilder.backend.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10
Environment=CUDA_VISIBLE_DEVICES=0

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable wakebuilder
sudo systemctl start wakebuilder
```

---

## Checklist

Before going to production:

- [ ] HTTPS configured with valid certificate
- [ ] CORS restricted to allowed origins
- [ ] Authentication implemented
- [ ] File upload limits set
- [ ] Monitoring configured
- [ ] Backup procedures tested
- [ ] Resource limits defined
- [ ] Log rotation configured
- [ ] Firewall rules applied
- [ ] Restart procedures documented
