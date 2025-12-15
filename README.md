# LLM Evaluation Pipeline

Production-ready pipeline for evaluating AI chatbot responses against ground truth context from vector databases. Designed to scale to millions of daily conversations with minimal latency and cost.

##  Features

- **Robust JSON Parsing**: Handles malformed JSON with multiple fallback strategies
- **Multiple Evaluation Metrics**:
  - Response Relevance (0-10)
  - Completeness (0-10)
  - Hallucination Detection (0-10, inverse scoring)
  - Factual Accuracy (0-10)
- **Production-Ready Scalability**:
  - Parallel processing with ThreadPoolExecutor
  - Context caching to reduce redundant operations
  - Batch processing for memory efficiency
  - Progressive result saving
- **Flexible LLM Backend**:
  - Default: Ollama (free, local inference)
  - Optional: Anthropic Claude API
- **Real-time Monitoring**: Progress tracking and detailed logging

##  Requirements

- Python 3.8+
- Ollama installed and running (for local evaluation)
- OR Anthropic API key (for Claude-based evaluation)

##  Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd llm-evaluation-pipeline

# Create virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Ollama (Recommended for Free Evaluation)

```bash
# Install Ollama from https://ollama.ai

# Start Ollama server
ollama serve

# Pull Mistral model (in another terminal)
ollama pull mistral:7b
```

### 3. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings
# For Ollama (default):
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=mistral:7b

# For Anthropic Claude (optional):
# ANTHROPIC_API_KEY=your_api_key_here
```

### 4. Run Evaluation

**Single Conversation:**
```bash
python evaluator.py json_files/sample_context_vectors-01.json json_files/sample-chat-conversation-01.json
```

**Batch Processing (Production):**
```bash
python batch_evaluator.py ./conversations ./vectors ./results 8
```

## Input File Structure

### Vector Context File (`sample_context_vectors-01.json`)
```json
{
  "status": "success",
  "data": {
    "vector_data": [
      {
        "id": 123,
        "text": "Context text from knowledge base...",
        "source_url": "https://example.com/article"
      }
    ]
  }
}
```

### Chat Conversation File (`sample-chat-conversation-01.json`)
```json
{
  "chat_id": 78128,
  "user_id": 77096,
  "conversation_turns": [
    {
      "turn": 1,
      "role": "User",
      "message": "User's question here",
      "created_at": "2025-11-16T17:04:44.000000Z"
    },
    {
      "turn": 2,
      "role": "AI/Chatbot",
      "message": "AI's response here",
      "created_at": "2025-11-16T17:04:50.000000Z"
    }
  ]
}
```

##  Output Format

```json
{
  "chat_id": "78128",
  "user_id": "77096",
  "timestamp": "2025-12-14T10:30:00",
  "model_used": "mistral:7b",
  "total_turns_evaluated": 5,
  "average_scores": {
    "relevance": 8.5,
    "completeness": 7.8,
    "hallucination": 9.2,
    "factual_accuracy": 8.0,
    "overall": 8.38
  },
  "total_latency_ms": 5230.45,
  "avg_latency_per_turn_ms": 1046.09,
  "total_tokens": 3421,
  "detailed_results": [
    {
      "turn": 1,
      "user_query": "User's question...",
      "ai_response": "AI's answer...",
      "evaluation": {
        "relevance_score": 9,
        "completeness_score": 8,
        "hallucination_score": 10,
        "factual_accuracy_score": 8,
        "issues": [],
        "missing_info": ["Additional details about X"],
        "hallucinations": [],
        "latency_ms": 1050.23,
        "input_tokens": 450,
        "output_tokens": 234
      }
    }
  ]
}
```

## Architecture

### System Design

```
┌─────────────────┐
│  Input Files    │
│  - Vectors JSON │
│  - Chat JSON    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  JSON Parser    │
│  - 4 fallback   │
│    strategies   │
│  - Error handle │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Extractors    │
│  - Conversation │
│  - Context      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│  LLM Evaluator  │◄────►│  Ollama API  │
│  - Scoring      │      │  or Claude   │
│  - Validation   │      └──────────────┘
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Result Saver   │
│  - JSON output  │
│  - Metrics      │
└─────────────────┘
```

### Key Components

1. **JSONParser**: Robust parser with 4 fallback strategies
   - Standard JSON parsing
   - Cleaned JSON with regex fixes
   - Object extraction from malformed files
   - UTF-8 BOM handling

2. **ConversationExtractor**: Flexible conversation pair extraction
   - Handles multiple data formats
   - Matches user queries with AI responses
   - Preserves metadata

3. **ContextExtractor**: Vector context aggregation
   - Supports multiple vector formats
   - Limits context size for performance
   - Deduplication

4. **LLMEvaluator**: Core evaluation engine
   - Structured prompting for consistent scoring
   - Response parsing with fallbacks
   - Token tracking and cost estimation

5. **BatchEvaluator**: Production scaling
   - Parallel processing (ThreadPoolExecutor)
   - Context caching
   - Progressive saving
   - Error recovery

##  Design Decisions

### Why This Architecture?

1. **Modular Design**
   - Each component is independent and testable
   - Easy to swap LLM backends (Ollama ↔ Claude)
   - Extensible for new evaluation metrics

2. **Robust JSON Parsing**
   - Real-world JSON files are often malformed
   - Multiple fallback strategies ensure high success rate
   - Prevents pipeline failures from data issues

3. **Flexible Data Extraction**
   - No hardcoded field names
   - Handles various API response formats
   - Graceful degradation when data is missing

4. **Performance Optimization**
   - Context caching reduces redundant vector DB queries
   - Parallel processing maximizes CPU utilization
   - Batch processing controls memory usage

5. **Production-Ready Features**
   - Progressive saving prevents data loss
   - Comprehensive logging for debugging
   - Error recovery and retry logic

### Alternative Approaches Considered

| Approach | Why Not Used |
|----------|--------------|
| **Hardcoded JSON paths** | Fails with format variations |
| **Single-threaded** | Too slow for millions of conversations |
| **In-memory processing** | Memory limits at scale |
| **Cloud LLM only** | High costs, latency, and API limits |
| **No caching** | Wasteful recomputation |

##  Scaling to Millions of Conversations

### Latency Optimization

1. **Parallel Processing**
   - ThreadPoolExecutor for concurrent LLM calls
   - Default: 4 workers (configurable)
   - Typical speedup: 3-4x

2. **Context Caching**
   - Cache vector contexts by file hash
   - Reduces vector DB queries by ~90%
   - Memory-efficient with periodic clearing

3. **Batch Processing**
   - Process 100 conversations at a time
   - Clear caches between batches
   - Prevents memory bloat

4. **Progressive Saving**
   - Save results immediately after evaluation
   - Recover from failures without reprocessing
   - Enables distributed processing

### Cost Optimization

1. **Local Inference (Ollama)**
   - Zero API costs
   - Typical latency: 1-2s per evaluation
   - Recommended for high-volume processing

2. **Efficient Prompting**
   - Truncate context to 3000 chars
   - Structured JSON output format
   - Temperature = 0 for consistency

3. **Token Management**
   - Track input/output tokens
   - Calculate cost per evaluation
   - Budget monitoring

### Production Deployment Strategy

```
┌─────────────────────────────────────────┐
│  Message Queue (e.g., RabbitMQ, Kafka)  │
│  - Conversation events                   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Worker Pool (Kubernetes/Docker)         │
│  ├─ Worker 1: batch_evaluator.py        │
│  ├─ Worker 2: batch_evaluator.py        │
│  └─ Worker N: batch_evaluator.py        │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Results Database (PostgreSQL/MongoDB)  │
│  - Evaluation scores                     │
│  - Aggregated metrics                    │
└─────────────────────────────────────────┘
```

**Estimated Throughput:**
- Single worker: ~300-500 conversations/hour
- 10 workers: ~3,000-5,000 conversations/hour
- 100 workers: ~30,000-50,000 conversations/hour
- Can scale horizontally to millions per day

### Cost Estimates (at scale)

**Ollama (Local):**
- Cost: $0 (after hardware investment)
- Latency: 1-2s per evaluation
- Best for: High volume (>100K/day)

**Anthropic Claude:**
- Cost: ~$0.001-0.002 per evaluation
- Latency: 500ms-1s per evaluation
- Best for: Low-medium volume (<50K/day)

**1 Million conversations/day:**
- Ollama: $0/day
- Claude: $1,000-2,000/day

##  Configuration

### Environment Variables

```bash
# Ollama settings
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=mistral:7b           # or llama2:13b, mixtral:8x7b
OLLAMA_TIMEOUT=120                # seconds

# Anthropic settings (optional)
ANTHROPIC_API_KEY=sk-ant-xxx
EVALUATION_MODEL=claude           # uses Claude Sonnet 4

# Batch processing
MAX_WORKERS=4                     # parallel threads
BATCH_SIZE=100                    # conversations per batch
```

### Choosing the Right Model

| Model | Speed | Accuracy | Cost | Use Case |
|-------|-------|----------|------|----------|
| **mistral:7b** | Fast | Good | Free | High-volume production |
| **llama2:13b** | Medium | Better | Free | Balanced |
| **mixtral:8x7b** | Slow | Best | Free | Quality-critical |
| **Claude Sonnet** | Fast | Excellent | Paid | Low-volume, high-quality |

## Monitoring & Analytics

### Real-time Metrics

```bash
# View evaluation progress
tail -f evaluation.log

# Monitor Ollama performance
curl http://localhost:11434/api/ps
```

### Aggregate Analysis

```python
import json
import glob

# Load all results
results = []
for file in glob.glob("evaluation_results/*.json"):
    with open(file) as f:
        results.append(json.load(f))

# Calculate global averages
avg_relevance = sum(r['average_scores']['relevance'] for r in results) / len(results)
avg_hallucination = sum(r['average_scores']['hallucination'] for r in results) / len(results)

print(f"Global Avg Relevance: {avg_relevance:.2f}")
print(f"Global Avg Hallucination Score: {avg_hallucination:.2f}")
```

## Troubleshooting

### Common Issues

**1. "Cannot connect to Ollama"**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve
```

**2. "JSON parsing failed"**
- Check file encoding (should be UTF-8)
- Validate JSON at jsonlint.com
- The parser has fallbacks, but very corrupted files may still fail

**3. "Out of memory"**
```bash
# Reduce batch size
python batch_evaluator.py ./conversations ./vectors ./results 2
# ↑ Reduce max_workers from 4 to 2
```

**4. "Model not found"**
```bash
# Pull the model
ollama pull mistral:7b
```

## Testing

```bash
# Test with sample files
python evaluator.py json_files/sample_context_vectors-01.json json_files/sample-chat-conversation-01.json

# Expected output: evaluation_results_<id>_<timestamp>.json
```

## Future Enhancements

- [ ] Distributed processing with Celery/Redis
- [ ] Real-time dashboard with Grafana
- [ ] A/B testing framework for prompt optimization
- [ ] Integration with popular vector DBs (Pinecone, Weaviate)
- [ ] Automated retraining based on evaluation scores
- [ ] Support for multi-language evaluation

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details
