# Production-Ready STT Server

## Overview

This is a production-ready Speech-to-Text (STT) server designed to handle thousands of concurrent phone calls with zero hard-coded delays and self-healing capabilities.

## Key Features

### 🚀 High Performance
- **Zero blocking operations** - No hard-coded sleeps or delays
- **Async-first architecture** - Built on asyncio for maximum concurrency
- **Non-blocking error recovery** - Self-healing without performance impact
- **Connection pooling** - Efficient resource management

### 🔧 Self-Healing
- **Automatic error recovery** - Circuit breaker pattern implementation
- **Health monitoring** - Real-time health checks and metrics
- **Graceful degradation** - Continues serving other calls during issues
- **Resource cleanup** - Automatic memory leak prevention

### 📊 Production Monitoring
- **Real-time metrics** - Connection rates, error rates, processing stats
- **Health endpoints** - `/health` and `/metrics` for monitoring
- **Production logging** - Structured logging for observability
- **Performance tracking** - Audio processing rates and latency

## Architecture

### Core Components

1. **AsyncRecorderManager** - Manages the STT recorder with async operations
2. **ServerMetrics** - Tracks performance and health metrics
3. **RecorderHealth** - Monitors recorder state and recovery
4. **Connection Management** - Handles thousands of concurrent connections
5. **Production Monitoring** - Real-time health and performance monitoring

### Error Recovery Strategy

- **Circuit Breaker Pattern** - Prevents cascading failures
- **Exponential Backoff** - Smart retry logic without blocking
- **Automatic Recovery** - Self-healing without manual intervention
- **Graceful Degradation** - Continues serving healthy connections

## Usage

### Basic Startup

```bash
# Start with production defaults
stt-server --model large-v2 --enable_realtime_transcription

# Start with custom configuration
stt-server \
  --model large-v2 \
  --rt-model tiny \
  --enable_realtime_transcription \
  --batch_size 32 \
  --device cuda \
  --control_port 8011 \
  --data_port 8012
```

### Production Configuration

```bash
# High-throughput production setup
stt-server \
  --model large-v2 \
  --rt-model tiny \
  --enable_realtime_transcription \
  --batch_size 64 \
  --realtime_batch_size 32 \
  --device cuda \
  --compute_type float16 \
  --allowed_latency_limit 200 \
  --handle_buffer_overflow \
  --use_extended_logging
```

## Monitoring

### Health Check Endpoint

```bash
curl http://localhost:8080/health
```

Response:
```json
{
  "status": "healthy",
  "recorder_state": "ready",
  "uptime_seconds": 3600.5,
  "active_connections": 150,
  "total_connections": 5000,
  "connections_per_minute": 83.3,
  "audio_chunks_processed": 1500000,
  "transcription_errors": 5,
  "recorder_errors": 2,
  "last_activity_seconds": 0.1,
  "consecutive_errors": 0,
  "recovery_attempts": 0
}
```

### Metrics Endpoint

```bash
curl http://localhost:8080/metrics
```

Response:
```json
{
  "server": {
    "uptime_seconds": 3600.5,
    "start_time": 1640995200.0,
    "last_activity": 1640998800.5
  },
  "connections": {
    "active": 150,
    "total": 5000,
    "rate_per_minute": 83.3
  },
  "processing": {
    "audio_chunks_processed": 1500000,
    "transcription_errors": 5,
    "recorder_errors": 2
  },
  "recorder": {
    "state": "ready",
    "last_successful_processing": 1640998800.0,
    "consecutive_errors": 0,
    "recovery_attempts": 0,
    "can_recover": true
  }
}
```

## Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

EXPOSE 8011 8012 8080

CMD ["stt-server", "--model", "large-v2", "--enable_realtime_transcription"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: stt-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: stt-server
  template:
    metadata:
      labels:
        app: stt-server
    spec:
      containers:
      - name: stt-server
        image: stt-server:latest
        ports:
        - containerPort: 8011
        - containerPort: 8012
        - containerPort: 8080
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Load Balancer Configuration

```yaml
apiVersion: v1
kind: Service
metadata:
  name: stt-server-service
spec:
  selector:
    app: stt-server
  ports:
  - name: control
    port: 8011
    targetPort: 8011
  - name: data
    port: 8012
    targetPort: 8012
  - name: health
    port: 8080
    targetPort: 8080
  type: LoadBalancer
```

## Performance Tuning

### For High Concurrency

```bash
# Optimize for thousands of concurrent calls
stt-server \
  --batch_size 128 \
  --realtime_batch_size 64 \
  --allowed_latency_limit 500 \
  --handle_buffer_overflow \
  --compute_type float16 \
  --device cuda
```

### For Low Latency

```bash
# Optimize for real-time response
stt-server \
  --batch_size 16 \
  --realtime_batch_size 8 \
  --allowed_latency_limit 50 \
  --rt-model tiny \
  --compute_type float32
```

### For High Accuracy

```bash
# Optimize for transcription quality
stt-server \
  --model large-v2 \
  --rt-model base \
  --beam_size 10 \
  --beam_size_realtime 5 \
  --compute_type float32
```

## Troubleshooting

### Common Issues

1. **High Error Rate**
   - Check `/metrics` endpoint for error details
   - Monitor `consecutive_errors` and `recovery_attempts`
   - Verify GPU memory and compute resources

2. **Connection Drops**
   - Monitor `active_connections` vs `total_connections`
   - Check network connectivity and load balancer health
   - Verify client connection handling

3. **Performance Degradation**
   - Monitor `audio_chunks_processed` rate
   - Check `last_activity_seconds` for processing delays
   - Verify batch sizes and compute type settings

### Log Analysis

```bash
# Monitor production logs
tail -f /var/log/stt-server.log | grep "\[PRODUCTION\]"

# Check error rates
grep "ERROR" /var/log/stt-server.log | wc -l

# Monitor connection rates
grep "Control client connected" /var/log/stt-server.log | wc -l
```

## Scaling

### Horizontal Scaling

1. **Multiple Instances** - Deploy multiple server instances behind a load balancer
2. **Session Affinity** - Use sticky sessions for WebSocket connections
3. **Resource Isolation** - Separate instances for different workloads

### Vertical Scaling

1. **GPU Resources** - Increase GPU memory and compute units
2. **CPU Resources** - Scale CPU cores for audio processing
3. **Memory** - Increase RAM for batch processing

## Security

### Network Security

- Use TLS/SSL for WebSocket connections
- Implement authentication for control endpoints
- Use firewall rules to restrict access

### Resource Security

- Run with minimal required permissions
- Use container isolation
- Implement rate limiting per connection

## Monitoring and Alerting

### Prometheus Metrics

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'stt-server'
    static_configs:
      - targets: ['stt-server:8080']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### Grafana Dashboard

Create dashboards for:
- Connection rates and trends
- Error rates and types
- Audio processing performance
- Recorder health status
- Resource utilization

### Alerting Rules

```yaml
# alerting.yml
groups:
  - name: stt-server
    rules:
      - alert: HighErrorRate
        expr: transcription_errors + recorder_errors > 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
      
      - alert: RecorderUnhealthy
        expr: recorder_state != "ready"
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Recorder is unhealthy"
```

## Best Practices

1. **Resource Monitoring** - Monitor CPU, GPU, and memory usage
2. **Error Tracking** - Track and analyze error patterns
3. **Performance Testing** - Load test with realistic scenarios
4. **Backup and Recovery** - Implement backup strategies
5. **Documentation** - Keep deployment and configuration docs updated

## Support

For production issues:
1. Check the `/health` and `/metrics` endpoints
2. Review production logs for error patterns
3. Monitor resource utilization
4. Verify network connectivity
5. Check client connection handling 