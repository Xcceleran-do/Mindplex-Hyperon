# Deploying Mindplex-Hyperon to Google Cloud Platform

This guide will help you deploy the Mindplex-Hyperon API to Google Cloud Run.

## Prerequisites

1. **Google Cloud Account**: Create one at [cloud.google.com](https://cloud.google.com)
2. **GCP Project**: Create a new project or use an existing one
3. **gcloud CLI**: Install from [cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install)
4. **ASI API Key**: Your ASI1 API key (from .env file)

## Quick Deployment (Recommended)

### Step 1: Install gcloud CLI
```bash
# Follow instructions at: https://cloud.google.com/sdk/docs/install
# Then authenticate:
gcloud auth login
```

### Step 2: Set Your Environment Variables
```bash
# Set your GCP project ID
export PROJECT_ID="your-gcp-project-id"

# Set your ASI API key (required for the app to work)
export ASI_API_KEY="your-asi-api-key"

# Optional: Set region (default is us-central1)
export REGION="us-central1"
```

### Step 3: Deploy Using the Script
```bash
# Run the deployment script
./deploy.sh $PROJECT_ID $REGION
```

The script will:
- Enable required GCP APIs
- Build your Docker container
- Deploy to Cloud Run
- Return your service URL

## Manual Deployment

If you prefer to deploy manually:

### Step 1: Set Up GCP Project
```bash
# Set your project
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

### Step 2: Build Container
```bash
# Build and push the container image
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/mindplex-hyperon-api
```

### Step 3: Deploy to Cloud Run
```bash
gcloud run deploy mindplex-hyperon-api \
  --image gcr.io/YOUR_PROJECT_ID/mindplex-hyperon-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 3600 \
  --set-env-vars ASI_API_KEY="your-asi-api-key"
```

## Configuration Options

### Resource Allocation
- **Memory**: 2Gi (adjust based on your needs)
- **CPU**: 2 cores
- **Timeout**: 3600s (1 hour for long-running mining jobs)

### Environment Variables
Set these when deploying:
- `ASI_API_KEY`: Your ASI1 API key (required)

### Authentication
- Currently set to `--allow-unauthenticated` for easy access
- For production, consider adding authentication:
  ```bash
  # Remove --allow-unauthenticated and use:
  --no-allow-unauthenticated
  ```

## Continuous Deployment with Cloud Build

For automated deployments when you push to GitHub:

### Step 1: Connect GitHub Repository
```bash
# Go to Cloud Console > Cloud Build > Triggers
# Connect your GitHub repository
```

### Step 2: Create Build Trigger
- Trigger type: Push to branch
- Branch: `^ASI-main$`
- Build configuration: `cloudbuild.yaml`

### Step 3: Add Secret Manager for API Key
```bash
# Store your ASI API key in Secret Manager
echo -n "your-asi-api-key" | gcloud secrets create asi-api-key --data-file=-

# Grant Cloud Build access
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
gcloud secrets add-iam-policy-binding asi-api-key \
  --member="serviceAccount:$PROJECT_NUMBER@cloudbuild.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

## Testing Your Deployment

Once deployed, test your API:

```bash
# Get your service URL
SERVICE_URL=$(gcloud run services describe mindplex-hyperon-api \
  --platform managed \
  --region us-central1 \
  --format 'value(status.url)')

# Test health endpoint
curl $SERVICE_URL/api/health

# Test mining endpoint
curl -X POST $SERVICE_URL/api/mine \
  -H "Content-Type: application/json" \
  -d '{"conjunction_count": 3}'

# Test chat endpoint
curl -X POST $SERVICE_URL/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Mine rules with 3 patterns", "session_id": "test"}'
```

## Updating the Deployment

To update your deployment after making changes:

```bash
# Simple re-run of the deploy script
./deploy.sh $PROJECT_ID $REGION
```

## Monitoring and Logs

View logs in real-time:
```bash
gcloud run services logs tail mindplex-hyperon-api --region us-central1
```

Or view in Cloud Console:
- Go to Cloud Run > mindplex-hyperon-api > Logs

## Cost Optimization

Cloud Run pricing:
- **Free tier**: 2 million requests/month
- **Pay-per-use**: Only charged when handling requests
- Estimated cost for moderate usage: $10-50/month

To reduce costs:
- Reduce memory/CPU if not needed
- Set min instances to 0 (default)
- Monitor usage in Cloud Console

## Troubleshooting

### Build Fails
- Check `cloudbuild.yaml` configuration
- Verify all dependencies in `requirements.txt`
- Check build logs: `gcloud builds list`

### Deployment Fails
- Check service logs: `gcloud run services logs tail`
- Verify environment variables are set
- Check IAM permissions

### API Not Working
- Verify ASI_API_KEY is set correctly
- Check CORS settings in Flask app
- Review application logs

## Domain Mapping (Optional)

To use a custom domain:

```bash
gcloud run domain-mappings create \
  --service mindplex-hyperon-api \
  --domain api.yourdomain.com \
  --region us-central1
```

Then update your DNS records as instructed.

## Security Best Practices

1. **API Keys**: Use Secret Manager instead of environment variables
2. **Authentication**: Enable Cloud IAM for production
3. **HTTPS**: Cloud Run provides HTTPS by default
4. **Rate Limiting**: Consider Cloud Armor for DDoS protection
5. **CORS**: Configure appropriately for your frontend domain

## Support

For issues:
- Check Cloud Run docs: [cloud.google.com/run/docs](https://cloud.google.com/run/docs)
- View logs in Cloud Console
- Check this repository's issues

## Next Steps

After deployment:
1. Update your frontend to point to the new API URL
2. Set up monitoring and alerting
3. Configure CI/CD for automatic deployments
4. Consider setting up a staging environment
