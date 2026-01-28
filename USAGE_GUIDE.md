# Usage Guide for RAG Evaluation Framework

This guide provides detailed instructions on how to use the RAG Evaluation Framework to evaluate your RAG models using both rule-based and advanced LLM-based metrics.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Choosing an Evaluator](#choosing-an-evaluator)
3. [Evaluation Metrics](#evaluation-metrics)
4. [Data Preparation](#data-preparation)
5. [Using the Framework](#using-the-framework)
6. [Command-Line Tool](#command-line-tool)
7. [Interpreting Results](#interpreting-results)

## Quick Start

### Installation

```bash
git clone https://github.com/RESHAPELab/RAG-evaluation.git
cd RAG-evaluation

# Install dependencies (includes ragas for LLM-based evaluation)
pip install -r requirements.txt

# For optional features (Excel support):
pip install openpyxl
```

### Basic Usage (Rule-based Evaluator)

The basic evaluator requires no external API keys and uses rule-based metrics:

```python
from rag_evaluation import RAGEvaluator

evaluator = RAGEvaluator()

results = evaluator.evaluate(
    query="What is machine learning?",
    context="Machine learning is a subset of AI...",
    answer="ML allows computers to learn from data...",
    ground_truth="ML is a subset of AI..."
)

print(results)
```

### Advanced Usage (Ragas Evaluator)

The ragas evaluator provides LLM-based metrics for more sophisticated evaluation. Requires an OpenAI API key:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

```python
from rag_evaluation import RagasEvaluator

evaluator = RagasEvaluator()

results = evaluator.evaluate(
    query="What is machine learning?",
    context="Machine learning is a subset of AI...",
    answer="ML allows computers to learn from data...",
    ground_truth="ML is a subset of AI..."
)

print(results)
```

## Choosing an Evaluator

### Basic Evaluator (RAGEvaluator)

**Use when:**
- You want fast, rule-based evaluation
- No external API dependencies are desired
- You need reproducible, deterministic scores
- Cost is a concern (no API costs)

**Pros:**
- No API key required
- Fast execution
- No external dependencies
- Deterministic results

**Cons:**
- Less nuanced evaluation
- May miss semantic similarities
- Keyword-based approach has limitations

### Ragas Evaluator (RagasEvaluator)

**Use when:**
- You need sophisticated, LLM-based evaluation
- Semantic understanding is important
- You want state-of-the-art evaluation metrics
- You have an OpenAI API key

**Pros:**
- More accurate and nuanced evaluation
- Understands semantic similarity
- Detects hallucinations more effectively
- State-of-the-art metrics

**Cons:**
- Requires OpenAI API key
- Costs money per evaluation
- Slower than rule-based evaluation
- Non-deterministic (LLM-based)

## Evaluation Metrics

### Basic Evaluator Metrics (Rule-based)

#### 1. Faithfulness

**What it measures:** Whether the answer is grounded in the provided context rather than hallucinated.

**How it works:**
- Splits the answer into individual sentences
- Checks if each sentence has support in the context
- Uses keyword overlap to determine grounding

**Score interpretation:**
- 1.0: All statements are supported by context (excellent)
- 0.8-0.9: Most statements supported (very good)
- 0.6-0.7: Majority supported (good)
- 0.4-0.5: Half supported (fair)
- < 0.4: Mostly unsupported (poor)

**When to use:** Always use this metric to detect hallucinations.

### 2. Context Precision

**What it measures:** How much of the answer content comes from the ground truth context.

**How it works:**
- Extracts key terms from answer and ground truth
- Calculates overlap between terms
- Measures precision of information sourcing

**Score interpretation:**
- 1.0: Answer entirely based on ground truth (perfect)
- 0.8-0.9: Strong alignment with ground truth
- 0.6-0.7: Good alignment
- 0.4-0.5: Moderate alignment
- < 0.4: Weak alignment

**When to use:** Use when you have ground truth answers and want to ensure model uses correct information sources.

**Note:** Requires ground_truth parameter.

### 3. Relevance

**What it measures:** Whether the response is relevant to both the query and context.

**How it works:**
- Extracts key terms from query, answer, and context
- Calculates query-answer overlap (70% weight)
- Calculates context-answer overlap (30% weight)
- Combines scores for overall relevance

**Score interpretation:**
- 1.0: Highly relevant to both query and context
- 0.8-0.9: Very relevant
- 0.6-0.7: Relevant
- 0.4-0.5: Somewhat relevant
- < 0.4: Not relevant

**When to use:** Always use to ensure answers address the user's question.

### Ragas Evaluator Metrics (LLM-based)

The ragas evaluator uses advanced Large Language Models to evaluate RAG systems. These metrics provide more nuanced and accurate assessments compared to rule-based approaches.

#### 1. Faithfulness

**What it measures:** Factual consistency of the answer with the retrieved context.

**How it works:**
- Uses LLM to extract claims from the answer
- Verifies each claim against the context using LLM
- Calculates the ratio of supported claims

**Score interpretation:**
- 1.0: All claims are factually consistent (excellent)
- 0.8-0.9: Most claims are supported (very good)
- 0.6-0.7: Majority supported (good)
- < 0.6: Significant hallucination issues

**Advantages over basic:** 
- Understands semantic meaning, not just keywords
- Better at detecting subtle hallucinations
- Considers context and nuance

#### 2. Answer Relevancy

**What it measures:** How relevant the answer is to the user's query.

**How it works:**
- Uses LLM to assess semantic relevance
- Considers whether answer directly addresses the question
- Evaluates completeness of the response

**Score interpretation:**
- 1.0: Perfectly relevant and complete (excellent)
- 0.8-0.9: Highly relevant (very good)
- 0.6-0.7: Relevant but may miss some aspects (good)
- < 0.6: Answer doesn't adequately address query

**Advantages over basic:**
- Semantic understanding vs keyword matching
- Detects incomplete or off-topic answers
- Considers query intent

#### 3. Context Precision

**What it measures:** How relevant the retrieved context is to answering the query.

**How it works:**
- Uses LLM to evaluate if context contains information needed to answer query
- Measures signal-to-noise ratio in retrieved context
- Helps evaluate retrieval quality

**Score interpretation:**
- 1.0: Context is highly relevant and precise (excellent)
- 0.8-0.9: Context is relevant (very good)
- 0.6-0.7: Some relevant information (good)
- < 0.6: Context contains mostly irrelevant information

**Use case:** Evaluate and improve your retrieval system

#### 4. Context Recall

**What it measures:** Whether all necessary information to answer the query is in the retrieved context.

**How it works:**
- Compares ground truth answer with retrieved context
- Uses LLM to check if context contains all required information
- Identifies gaps in retrieval

**Score interpretation:**
- 1.0: All necessary information retrieved (excellent)
- 0.8-0.9: Most information present (very good)
- 0.6-0.7: Key information present but incomplete (good)
- < 0.6: Significant information missing

**Note:** Requires ground_truth parameter

**Use case:** Identify if your retrieval system is missing important information

## Data Preparation

### Supported Formats

#### 1. CSV Format

Create a CSV file with these columns:

```csv
query,context,answer,ground_truth
"What is ML?","ML is...","ML allows...","ML is..."
```

#### 2. JSON Format

```json
[
  {
    "query": "What is ML?",
    "context": "ML is...",
    "answer": "ML allows...",
    "ground_truth": "ML is..."
  }
]
```

#### 3. BibTeX Format (Jabref)

```bibtex
@article{key,
  title={Paper Title},
  abstract={Context text here...}
}
```

The loader will use:
- `title` as query
- `abstract` or `note` as context
- `abstract` as ground truth

### Data Requirements

**Required fields:**
- `query`: The user's question
- `context`: Retrieved context for the answer
- `answer`: The RAG model's generated answer

**Optional fields:**
- `ground_truth`: Reference answer (required for context_precision metric)

## Using the Framework

### Single Evaluation

#### Using Basic Evaluator

```python
from rag_evaluation import RAGEvaluator

evaluator = RAGEvaluator()

results = evaluator.evaluate(
    query="Your question here",
    context="Retrieved context here",
    answer="Generated answer here",
    ground_truth="Optional reference answer"
)

# Access scores
faithfulness_score = results['faithfulness']['score']
relevance_score = results['relevance']['score']
```

#### Using Ragas Evaluator

```python
from rag_evaluation import RagasEvaluator
import os

# Set API key (if not already set in environment)
os.environ['OPENAI_API_KEY'] = 'your-key-here'

evaluator = RagasEvaluator()

results = evaluator.evaluate(
    query="Your question here",
    context="Retrieved context here",
    answer="Generated answer here",
    ground_truth="Optional reference answer"
)

# Access scores
faithfulness_score = results['faithfulness']['score']
answer_relevancy_score = results['answer_relevancy']['score']
```

### Batch Evaluation

#### Using Basic Evaluator

```python
from rag_evaluation import RAGEvaluator

evaluator = RAGEvaluator()

results = evaluator.evaluate_batch(
    queries=["Q1", "Q2", "Q3"],
    contexts=["C1", "C2", "C3"],
    answers=["A1", "A2", "A3"],
    ground_truths=["GT1", "GT2", "GT3"]
)

# Get average scores
avg_scores = evaluator.get_average_scores(results)
```

#### Using Ragas Evaluator

```python
from rag_evaluation import RagasEvaluator

evaluator = RagasEvaluator()

# Ragas provides progress bar for batch evaluation
results = evaluator.evaluate_batch(
    queries=["Q1", "Q2", "Q3"],
    contexts=["C1", "C2", "C3"],
    answers=["A1", "A2", "A3"],
    ground_truths=["GT1", "GT2", "GT3"]
)

avg_scores = evaluator.get_average_scores(results)
```

### Loading from Files

Both evaluators work with the same data loading interface:

```python
from rag_evaluation import RAGEvaluator, RagasEvaluator
from rag_evaluation.data_ingestion import DataTableLoader

# Load data
loader = DataTableLoader()
data = loader.load_for_evaluation('my_data.csv')

# Evaluate with basic evaluator
evaluator = RAGEvaluator()
results = evaluator.evaluate_batch(**data)

# Or use ragas evaluator
ragas_evaluator = RagasEvaluator()
results = ragas_evaluator.evaluate_batch(**data)
```

### Custom Metric Selection

#### Basic Evaluator

```python
# Only evaluate faithfulness and relevance
evaluator = RAGEvaluator(metrics=['faithfulness', 'relevance'])
results = evaluator.evaluate(query, context, answer)
```

#### Ragas Evaluator

```python
# Only evaluate specific ragas metrics
evaluator = RagasEvaluator(metrics=['faithfulness', 'answer_relevancy'])
results = evaluator.evaluate(query, context, answer)
```

### Comparing Evaluators

You can use both evaluators to compare results:

```python
from rag_evaluation import RAGEvaluator, RagasEvaluator

# Prepare data
query = "What is machine learning?"
context = "Machine learning is a subset of AI..."
answer = "ML is a type of AI that learns from data."

# Basic evaluation
basic_eval = RAGEvaluator()
basic_results = basic_eval.evaluate(query, context, answer)
print("Basic scores:", {k: v['score'] for k, v in basic_results.items()})

# Ragas evaluation (requires API key)
ragas_eval = RagasEvaluator()
ragas_results = ragas_eval.evaluate(query, context, answer)
print("Ragas scores:", {k: v['score'] for k, v in ragas_results.items()})
```

## Command-Line Tool

The framework includes a command-line tool for easy evaluation:

### Basic Usage

```bash
python examples/evaluate.py data.csv
```

### With Verbose Output

```bash
python examples/evaluate.py data.json --verbose
```

### Save Results to File

```bash
python examples/evaluate.py data.csv --output results.json
```

### Specify Data Format

```bash
python examples/evaluate.py data.bib --type bibtex
```

### Select Specific Metrics

```bash
python examples/evaluate.py data.csv --metrics faithfulness relevance
```

### All Options

```bash
python examples/evaluate.py --help
```

## Interpreting Results

### Understanding Scores

Each metric returns a dictionary with:
- `score`: Float between 0.0 and 1.0
- `details`: Additional information about the evaluation

Example result:
```python
{
    'faithfulness': {
        'score': 0.857,
        'details': {
            'total_sentences': 3,
            'supported_sentences': 2,
            'unsupported_sentences': ['One unsupported sentence'],
            'reasoning': '2 out of 3 sentences are grounded in context'
        }
    }
}
```

### Score Thresholds

General guidelines:

| Score Range | Rating | Action |
|-------------|--------|--------|
| 0.8 - 1.0 | Excellent | Model performing well |
| 0.6 - 0.8 | Good | Minor improvements possible |
| 0.4 - 0.6 | Fair | Review model and data |
| 0.0 - 0.4 | Poor | Significant issues, needs attention |

### Common Issues and Solutions

#### Low Faithfulness Score
**Problem:** Model is hallucinating information

**Solutions:**
- Improve context quality
- Use stricter generation parameters
- Add explicit grounding instructions
- Reduce model temperature

#### Low Context Precision
**Problem:** Model not using correct information sources

**Solutions:**
- Improve retrieval system
- Better context ranking
- Train model on better examples
- Verify ground truth quality

#### Low Relevance Score
**Problem:** Answers don't address the question

**Solutions:**
- Improve query understanding
- Better prompt engineering
- Ensure context is relevant to query
- Check if query and answer are aligned

## Best Practices

1. **Always evaluate faithfulness** to catch hallucinations
2. **Use ground truth** when available for context precision
3. **Run batch evaluations** to get reliable average scores
4. **Keep data consistent** with your production use case
5. **Track metrics over time** to monitor improvements
6. **Combine with human evaluation** for best results

## Examples

See the `examples/` directory for complete code examples:

- `basic_usage.py`: Comprehensive examples
- `evaluate.py`: Command-line evaluation tool
- `sample_data.csv`: Example CSV data
- `sample_data.json`: Example JSON data
- `sample_data.bib`: Example BibTeX data

## Support

For issues or questions, please open an issue on GitHub:
https://github.com/RESHAPELab/RAG-evaluation/issues
