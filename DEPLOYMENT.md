# 🚀 Deployment Guide - Government Decision Support RAG System

## Quick Start (Local Development)

### 1. Environment Setup
```bash
# Clone and navigate to project
cd /path/to/RAG_MODEL

# Copy environment template
cp .env.example .env

# Edit .env file with your Google API key
nano .env
# Add: GOOGLE_API_KEY=your_gemini_api_key_here
```

### 2. Install Dependencies
```bash
# Install Python packages
pip install -r requirements.txt

# Or using conda
conda create -n gov-rag python=3.9
conda activate gov-rag
pip install -r requirements.txt
```

### 3. Run the Application
```bash
# Start the Streamlit app
streamlit run app_multilang.py

# Access at: http://localhost:8501
```

## 🐳 Docker Deployment

### Single Container
```bash
# Build the image
docker build -t gov-rag-system .

# Run with environment file
docker run -p 8501:8501 --env-file .env -v $(pwd)/data:/app/data gov-rag-system
```

### Docker Compose (Recommended)
```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Production with Nginx
```bash
# Start with production profile
docker-compose --profile production up -d

# This includes:
# - Application container
# - Nginx reverse proxy
# - SSL termination
# - Health checks
```

## ☸️ Kubernetes Deployment

### Prerequisites
- Kubernetes cluster (1.20+)
- kubectl configured
- Cert-manager (for SSL)
- Ingress controller

### Deploy to Kubernetes
```bash
# Apply the configuration
kubectl apply -f k8s-deployment.yaml

# Check deployment status
kubectl get pods -n gov-rag-system
kubectl get services -n gov-rag-system
kubectl get ingress -n gov-rag-system

# View logs
kubectl logs -f deployment/gov-rag-deployment -n gov-rag-system
```

### Configuration Updates
```bash
# Update API key
kubectl create secret generic api-keys \
  --from-literal=google-api-key=your_new_key \
  --namespace=gov-rag-system \
  --dry-run=client -o yaml | kubectl apply -f -

# Update configuration
kubectl patch configmap gov-rag-config -n gov-rag-system --patch '{"data":{"LLM_TEMPERATURE":"0.2"}}'

# Restart deployment
kubectl rollout restart deployment/gov-rag-deployment -n gov-rag-system
```

## 🔧 Configuration Options

### Environment Variables
```env
# Core Configuration
GOOGLE_API_KEY=your_api_key              # Required: Google AI API key
DOMAIN=Local Government Planning         # System domain
DEFAULT_LANGUAGE=English                 # Default UI language

# AI Configuration  
LLM_MODEL=gemini-pro                     # LLM model name
LLM_TEMPERATURE=0.1                      # Response randomness (0.0-1.0)
EMBEDDING_MODEL=models/embedding-001     # Embedding model

# Retrieval Configuration
CHUNK_SIZE=1200                          # Document chunk size
CHUNK_OVERLAP=300                        # Chunk overlap
RETRIEVAL_K=6                            # Documents to retrieve
RETRIEVAL_FETCH_K=20                     # Candidates to fetch

# Security Configuration
AUDIT_LOGGING=true                       # Enable audit logging
INPUT_SANITIZATION=true                  # Enable input sanitization
SENSITIVE_CONTENT_DETECTION=true         # Detect sensitive content

# Performance Configuration
STREAMLIT_SERVER_PORT=8501               # Server port
STREAMLIT_SERVER_ADDRESS=0.0.0.0         # Server address
```

## 📊 Monitoring & Maintenance

### Health Checks
```bash
# Application health
curl http://localhost:8501/_stcore/health

# Docker health check
docker inspect gov-rag-system | jq '.[0].State.Health'

# Kubernetes health
kubectl get pods -n gov-rag-system -o wide
```

### Log Management
```bash
# View application logs
tail -f gov_rag_audit.log

# Docker logs
docker-compose logs -f gov-rag-system

# Kubernetes logs
kubectl logs -f deployment/gov-rag-deployment -n gov-rag-system
```

### Performance Monitoring
```bash
# Monitor resource usage
docker stats gov-rag-system

# Kubernetes metrics
kubectl top pods -n gov-rag-system
kubectl describe pod -n gov-rag-system
```

## 🔒 Security Configuration

### SSL/TLS Setup
```bash
# Generate self-signed certificate for development
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# For production, use Let's Encrypt with cert-manager
kubectl apply -f https://github.com/jetstack/cert-manager/releases/download/v1.8.0/cert-manager.yaml
```

### Network Security
```bash
# Configure firewall (Ubuntu/Debian)
sudo ufw allow 8501/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Docker network isolation
docker network create gov-rag-network --driver bridge
```

### Data Security
```bash
# Encrypt data at rest
echo "your_encryption_key" > encryption.key
chmod 600 encryption.key

# Backup vector store
tar -czf vector_store_backup_$(date +%Y%m%d).tar.gz vector_store/
```

## 📈 Scaling Configuration

### Horizontal Scaling
```bash
# Docker Compose scaling
docker-compose up -d --scale gov-rag-system=3

# Kubernetes scaling
kubectl scale deployment gov-rag-deployment --replicas=5 -n gov-rag-system
```

### Load Balancing
```yaml
# Add to docker-compose.yml
nginx:
  image: nginx:alpine
  ports:
    - "80:80"
  volumes:
    - ./nginx-lb.conf:/etc/nginx/nginx.conf
  depends_on:
    - gov-rag-system
```

### Database Optimization
```python
# Vector store optimization
# Adjust in app_multilang.py
CHROMA_SETTINGS = {
    "anonymized_telemetry": False,
    "allow_reset": True,
    "max_batch_size": 100,
    "embedding_threads": 4
}
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. API Key Issues
```bash
# Verify API key
curl -H "Content-Type: application/json" \
     -H "Authorization: Bearer $GOOGLE_API_KEY" \
     https://generativelanguage.googleapis.com/v1/models
```

#### 2. Memory Issues
```bash
# Increase Docker memory limits
docker run -p 8501:8501 --memory=4g --cpus="2" gov-rag-system

# Monitor memory usage
docker stats --no-stream
```

#### 3. Vector Store Issues
```bash
# Rebuild vector store
rm -rf vector_store/
# Restart application - it will rebuild automatically
```

#### 4. Port Conflicts
```bash
# Find process using port
lsof -i :8501

# Use alternative port
streamlit run app_multilang.py --server.port=8502
```

### Debug Mode
```bash
# Enable debug logging
export STREAMLIT_LOGGER_LEVEL=debug
export LANGCHAIN_DEBUG=true

# Run with verbose output
streamlit run app_multilang.py --logger.level=debug
```

## 📋 Maintenance Tasks

### Regular Maintenance
```bash
# Weekly: Update vector store
# Monthly: Update dependencies
# Quarterly: Security audit
# Annually: Performance review
```

### Backup Strategy
```bash
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/gov-rag-$DATE"

mkdir -p $BACKUP_DIR
tar -czf "$BACKUP_DIR/data.tar.gz" data/
tar -czf "$BACKUP_DIR/vector_store.tar.gz" vector_store/
cp .env "$BACKUP_DIR/"
cp gov_rag_audit.log "$BACKUP_DIR/"

echo "Backup completed: $BACKUP_DIR"
```

### Update Process
```bash
# 1. Backup current system
./backup.sh

# 2. Pull updates
git pull origin main

# 3. Update dependencies
pip install -r requirements.txt --upgrade

# 4. Rebuild vector store if needed
rm -rf vector_store/

# 5. Test system
python3 -c "import app_multilang; print('Import successful')"

# 6. Restart services
docker-compose down && docker-compose up -d
```

## 📞 Support & Contact

### Technical Support
- **Email**: techsupport@government.local
- **Phone**: +1 (555) 123-TECH
- **Hours**: Monday-Friday, 8 AM - 6 PM

### Emergency Contacts
- **System Admin**: +1 (555) 911-TECH
- **Security Team**: +1 (555) 911-SEC
- **On-call Support**: Available 24/7

### Documentation
- **API Docs**: `/docs` endpoint when running
- **User Guide**: See README.md
- **Architecture**: See system diagrams in /docs
- **Security**: See security policy in /security

---

**Last Updated**: September 2025  
**Version**: 2.0  
**Maintained by**: Government IT Department
