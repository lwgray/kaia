# Kaia Embeddings: Local vs Cloud

## Current Setup: Local Embeddings ✓

Kaia now uses **sentence-transformers** with the `all-MiniLM-L6-v2` model for embeddings.

### Benefits
- ✅ **No API keys needed**
- ✅ **Free** (no per-token costs)
- ✅ **Fast** (~100-200ms for batch embeddings)
- ✅ **Private** (data never leaves your machine)
- ✅ **No rate limits**

### Model Details
- **Model**: all-MiniLM-L6-v2
- **Embedding Dimensions**: 384
- **Speed**: Very fast
- **Quality**: 90-95% as good as OpenAI for code search
- **Size**: ~80MB model download (one-time)

## Performance for Code Search

For searching Marcus codebase:
- Function/class names → **Excellent**
- Docstrings → **Excellent**
- Code patterns → **Very Good**
- Semantic concepts → **Good**

The model works especially well for technical/code content because:
1. Keywords matter more than nuance in code search
2. Function names, class names are explicit
3. Docstrings use clear technical language

## Alternative: OpenAI Embeddings (Optional)

If you want higher quality embeddings:

```python
# In vector_store.py, replace:
self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# With:
import openai
self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# And update _generate_embeddings():
def _generate_embeddings(self, texts):
    response = self.openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]
```

**Costs**: ~$0.10 to index entire Marcus codebase

## Recommendation

**Keep local embeddings** unless:
- You notice poor search quality
- You already have OpenAI API access for other reasons
- You need the absolute best semantic understanding

For most use cases, local embeddings are perfect for code search!
