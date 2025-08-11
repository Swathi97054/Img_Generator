# Advanced Text-to-Image Generation Service

A sophisticated, containerized AI service that generates high-quality images from text descriptions using state-of-the-art diffusion models. This service includes advanced safety features, multi-language support, LoRA model integration, and comprehensive monitoring capabilities.

## 🚀 Features

### Core Capabilities
- **High-Quality Image Generation**: Uses Stable Diffusion and Stable Diffusion XL models
- **Multi-Language Support**: Supports English, Spanish, French, German, and Italian with language-specific prompt enhancement
- **LoRA Integration**: Support for custom LoRA models with trigger words and scaling
- **Template System**: Pre-built templates for portraits, landscapes, products, and anime styles
- **Seed Control**: Reproducible image generation with customizable random seeds

### Advanced Safety & Security
- **Content Safety Filtering**: Multi-category safety checks (violence, hate speech, NSFW content, etc.)
- **Intention Classification**: AI-powered analysis of user prompts for educational, artistic, or harmful intent
- **Multi-Language Safety**: Safety checks in supported languages
- **Audit Logging**: Comprehensive logging of all requests and safety violations
- **Rate Limiting**: Built-in protection against abuse
- **API Key Authentication**: Secure admin functions

### User Experience
- **Chat Memory**: Persistent user interaction history
- **Prompt Enhancement**: AI-powered prompt improvement suggestions
- **Image Regeneration**: Regenerate images with same or new seeds
- **Download Management**: Easy access to generated images
- **Progress Tracking**: Real-time generation progress updates

### Technical Features
- **Model Management**: Automatic model loading/unloading for memory optimization
- **Performance Monitoring**: Memory usage tracking and optimization
- **Comprehensive Logging**: Rotating log files with multiple log levels
- **Health Monitoring**: Service health checks and status endpoints
- **Docker Containerization**: Easy deployment and scaling

## 📋 Prerequisites

- **Docker** and **Docker Compose**
- **GPU Support** (recommended for optimal performance)
- **Sufficient Disk Space**: At least 10GB for models and generated images
- **Memory**: Minimum 8GB RAM, 16GB+ recommended for large models

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone ""
   cd t2i
   ```

2. **Create required directories**:
   ```bash
   mkdir -p models loras outputs logs
   ```

3. **Download AI models** (optional - will download automatically on first use):
   - Place Stable Diffusion models in the `models/` directory
   - Place LoRA models in the `loras/` directory

4. **Start the service**:
   ```bash
   docker-compose up -d
   ```

   The service will be available at `http://localhost:8000`

## 🎯 Usage

### Basic Image Generation

Generate an image with a simple prompt:
```bash
curl -X POST http://localhost:8000/generate-image/ \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "stabilityai/stable-diffusion-2-1",
    "prompt": "a beautiful mountain landscape at sunset",
    "language": "en",
    "num_inference_steps": 30,
    "guidance_scale": 7.5
  }'
```

### Advanced Generation with LoRA

Generate an image using a custom LoRA model:
```bash
curl -X POST http://localhost:8000/generate-image/ \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "stabilityai/stable-diffusion-2-1",
    "prompt": "a portrait of a woman",
    "use_lora": true,
    "lora_id": "my_custom_lora",
    "lora_scale": 0.8,
    "template_type": "portrait"
  }'
```

### Prompt Analysis and Enhancement

Analyze and enhance a prompt:
```bash
curl -X POST "http://localhost:8000/analyze-and-enhance-prompt/?prompt=a%20cat&language=en&apply_enhancements=true"
```

### Safety Check

Check if a prompt passes safety filters:
```bash
curl -X POST "http://localhost:8000/check-prompt-safety/?prompt=my%20prompt&language=en"
```

## 🔌 API Endpoints

### Core Generation
- `POST /generate-image/` - Main image generation endpoint
- `POST /generate-with-lora/` - LoRA-specific generation
- `GET /regenerate/{chat_id}` - Regenerate existing images

### Model Management
- `GET /models/` - List available models
- `GET /loras/` - List available LoRA models
- `GET /lora/{lora_id}` - Get LoRA model information

### Safety & Analysis
- `POST /check-prompt-safety/` - Safety content filtering
- `POST /check-prompt-intention/` - Intent classification
- `POST /analyze-and-enhance-prompt/` - Prompt improvement

### User Management
- `GET /user-history/{user_id}` - User interaction history
- `GET /similar-prompts/` - Find similar previous prompts
- `GET /history/{chat_id}` - Detailed interaction history

### System Status
- `GET /health/` - Service health check
- `GET /memory-stats/` - Memory usage statistics
- `GET /languages/` - Supported languages
- `GET /safety-categories/` - Available safety categories

### File Operations
- `GET /download/{chat_id}` - Download generated images
- `GET /` - Web interface (if enabled)

## ⚙️ Configuration

### Environment Variables

Configure the service behavior through environment variables in `docker-compose.yaml`:

```yaml
environment:
  - DIFFUSERS_PROGRESS_BAR=0          # Disable diffusion progress bars
  - TRANSFORMERS_PROGRESS_BAR=0       # Disable transformer progress bars
  - HF_HUB_DISABLE_PROGRESS_BARS=1   # Disable Hugging Face progress bars
  - TF_CPP_MIN_LOG_LEVEL=2           # TensorFlow logging level
```

### Resource Limits

Uncomment and adjust resource limits in `docker-compose.yaml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '2.0'      # CPU cores
      memory: 8G       # Memory limit
```

## 📁 Project Structure

```
t2i/
├── Dockerfile              # Container definition
├── docker-compose.yaml     # Docker Compose configuration
├── requirements.txt        # Python dependencies
├── text2img.py            # Main application (FastAPI)
├── models/                 # AI model storage
├── loras/                  # LoRA model storage
├── outputs/                # Generated images
└── logs/                   # Application logs
    ├── text2img.log       # Main application log
    ├── errors.log         # Error-specific log
    └── debug.log          # Debug information log
```

## 🔒 Safety Features

### Content Filtering
The service implements comprehensive content safety measures:

- **Violence Detection**: Identifies and blocks violent content
- **Hate Speech Filtering**: Prevents generation of discriminatory content
- **NSFW Content Control**: Configurable adult content filtering
- **Illegal Activity Prevention**: Blocks prompts related to illegal activities
- **Religious Sensitivity**: Respects religious and cultural boundaries

### Multi-Language Safety
Safety checks are performed in the user's language:
- English, Spanish, French, German, Italian
- Language-specific pattern recognition
- Cultural context awareness

### Admin Override
- API key-based admin functions
- Bypass safety checks when necessary
- Audit logging of all admin actions

## 🌐 Multi-Language Support

### Supported Languages
- **English (en)**: Full support with extensive prompt enhancement
- **Spanish (es)**: Spanish-specific style and cultural context
- **French (fr)**: French artistic and cultural elements
- **German (de)**: German language optimization
- **Italian (it)**: Italian cultural and artistic context

### Language Features
- Automatic prompt enhancement based on language
- Cultural context awareness
- Language-specific safety filtering
- Localized negative prompts

## 🎨 LoRA Model Support

### LoRA Features
- **Automatic Detection**: Scans `loras/` directory for available models
- **Trigger Words**: Automatic prompt enhancement with LoRA-specific triggers
- **Scaling Control**: Adjustable LoRA effect strength (0.0-1.0)
- **Model Information**: Detailed metadata for each LoRA model

### LoRA Management
- Hot-swappable LoRA models
- Memory-efficient loading/unloading
- Automatic trigger word detection
- Performance optimization

## 📊 Monitoring & Logging

### Log Levels
- **INFO**: General application events
- **DEBUG**: Detailed debugging information
- **ERROR**: Error conditions and exceptions
- **WARNING**: Warning conditions

### Log Rotation
- **Main Log**: 10MB with 5 backup files
- **Error Log**: 5MB with 3 backup files
- **Debug Log**: 10MB with 2 backup files

### Health Monitoring
- Service health checks
- Memory usage tracking
- Model loading status
- Performance metrics

## 🚨 Troubleshooting

### Common Issues

1. **Service won't start**:
   ```bash
   docker-compose logs text2img
   ```

2. **Out of memory errors**:
   - Reduce `num_inference_steps`
   - Use smaller models
   - Increase Docker memory limits

3. **Model loading issues**:
   - Verify model files in `models/` directory
   - Check file permissions
   - Ensure sufficient disk space

4. **Safety filter too strict**:
   - Adjust safety level parameters
   - Use admin API key for bypass (if authorized)

### Performance Optimization

- **GPU Acceleration**: Ensure CUDA support is available
- **Model Caching**: Keep frequently used models in memory
- **Batch Processing**: Process multiple requests efficiently
- **Memory Management**: Automatic model unloading for unused models

## 📈 Performance

### Generation Speed
- **Fast Mode**: 15-20 steps (~10-15 seconds)
- **Balanced Mode**: 30 steps (~20-30 seconds)
- **Quality Mode**: 50+ steps (~40-60 seconds)

### Memory Usage
- **Base Model**: 2-4GB VRAM
- **XL Model**: 6-8GB VRAM
- **LoRA Models**: +100-500MB per model

### Scalability
- Horizontal scaling with multiple containers
- Load balancing support
- Resource monitoring and optimization

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:
- Code style and standards
- Testing requirements
- Documentation updates
- Feature requests and bug reports

## 📄 License

[Your license information here]

## 🆘 Support

For support and questions:
- Check the troubleshooting section
- Review logs for error details
- Open an issue on GitHub
- Contact the development team

---

**Note**: This service includes advanced AI safety features and is designed for responsible use. Please ensure compliance with local laws and ethical guidelines when generating content.
