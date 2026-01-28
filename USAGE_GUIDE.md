# Usage Guide for RAG Evaluation Framework

This guide provides detailed instructions on how to use the RAG Evaluation Framework to evaluate your RAG models.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Evaluation Metrics](#evaluation-metrics)
3. [Data Preparation](#data-preparation)
4. [Using the Framework](#using-the-framework)
5. [Command-Line Tool](#command-line-tool)
6. [Interpreting Results](#interpreting-results)

## Quick Start

### Installation

The framework requires only Python 3.6+ with no external dependencies for basic functionality:

```bash
git clone https://github.com/RESHAPELab/RAG-evaluation.git
cd RAG-evaluation
```

For optional features (Excel support), install:
```bash
pip install openpyxl
```

### Basic Usage

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

## Evaluation Metrics

### 1. Faithfulness

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

### Batch Evaluation

```python
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

### Loading from Files

```python
from rag_evaluation import RAGEvaluator
from rag_evaluation.data_ingestion import DataTableLoader

# Load data
loader = DataTableLoader()
data = loader.load_for_evaluation('my_data.csv')

# Evaluate
evaluator = RAGEvaluator()
results = evaluator.evaluate_batch(**data)
```

### Custom Metric Selection

```python
# Only evaluate faithfulness and relevance
evaluator = RAGEvaluator(metrics=['faithfulness', 'relevance'])

results = evaluator.evaluate(query, context, answer)
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
