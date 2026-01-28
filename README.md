# RAG Evaluation Framework

A comprehensive framework for evaluating Retrieval-Augmented Generation (RAG) models using multiple metrics including faithfulness, context precision, and relevance.

## Features

- **Multiple Evaluation Metrics**:
  - **Faithfulness**: Checks if the answer is grounded in the provided context rather than hallucinated
  - **Context Precision**: Measures how much of the answer comes from the ground truth context
  - **Relevance**: Checks if the generated response is relevant to the query and context

- **Data Ingestion Support**:
  - Jabref (BibTeX) format
  - DataTable formats (CSV, JSON, Excel)

- **Easy-to-Use API**:
  - Single evaluation
  - Batch evaluation
  - Custom metric selection

## Installation

```bash
# Clone the repository
git clone https://github.com/RESHAPELab/RAG-evaluation.git
cd RAG-evaluation

# No external dependencies required for basic functionality
# Optional dependencies can be installed for enhanced features:
# pip install openpyxl  # For Excel support
# pip install bibtexparser  # For advanced BibTeX parsing
```

## Quick Start

### Single Evaluation

```python
from rag_evaluation import RAGEvaluator

# Initialize evaluator
evaluator = RAGEvaluator()

# Evaluate a single output
results = evaluator.evaluate(
    query="What is machine learning?",
    context="Machine learning is a subset of AI that enables systems to learn...",
    answer="Machine learning allows computers to learn from data...",
    ground_truth="Machine learning is a subset of AI..."
)

print(results)
```

### Batch Evaluation

```python
from rag_evaluation import RAGEvaluator

evaluator = RAGEvaluator()

# Evaluate multiple outputs
results = evaluator.evaluate_batch(
    queries=["What is ML?", "What is DL?"],
    contexts=["ML is...", "DL is..."],
    answers=["ML allows...", "DL uses..."],
    ground_truths=["ML is...", "DL is..."]
)

# Get average scores
avg_scores = evaluator.get_average_scores(results)
print(avg_scores)
```

### Loading Data from Files

```python
from rag_evaluation import RAGEvaluator
from rag_evaluation.data_ingestion import DataTableLoader, JabrefLoader

# Load from CSV
loader = DataTableLoader()
data = loader.load_for_evaluation('data.csv')

# Evaluate
evaluator = RAGEvaluator()
results = evaluator.evaluate_batch(**data)

# Load from Jabref/BibTeX
jabref_loader = JabrefLoader()
data = jabref_loader.load_for_evaluation('references.bib')
results = evaluator.evaluate_batch(**data)
```

## Evaluation Metrics

### Faithfulness

Evaluates whether the generated answer is grounded in the provided context and not hallucinated. 

- **Score Range**: 0.0 to 1.0
- **1.0**: All statements are supported by context
- **0.0**: No statements are supported by context

### Context Precision

Measures how much of the information in the answer comes from the ground truth context.

- **Score Range**: 0.0 to 1.0
- **1.0**: Answer is fully based on ground truth
- **0.0**: Answer has no overlap with ground truth

### Relevance

Checks if the generated response is relevant to both the query and the provided context.

- **Score Range**: 0.0 to 1.0
- **1.0**: Highly relevant answer
- **0.0**: Completely irrelevant answer

## Data Format

### CSV Format

```csv
query,context,answer,ground_truth
"What is ML?","ML is...","ML allows...","ML is..."
```

### JSON Format

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

### BibTeX Format (Jabref)

```bibtex
@article{key,
  title={Title},
  abstract={Context text...},
  note={Additional context...}
}
```

## Examples

Check the `examples/` directory for complete usage examples:

- `examples/basic_usage.py`: Comprehensive examples of all features
- `examples/sample_data.json`: Sample JSON data
- `examples/sample_data.csv`: Sample CSV data
- `examples/sample_data.bib`: Sample BibTeX data

Run the examples:

```bash
cd examples
python basic_usage.py
```

## Project Structure

```
rag_evaluation/
├── __init__.py           # Main package exports
├── evaluator.py          # RAGEvaluator class
├── metrics/
│   ├── __init__.py
│   ├── faithfulness.py   # Faithfulness metric
│   ├── context_precision.py  # Context precision metric
│   └── relevance.py      # Relevance metric
└── data_ingestion/
    ├── __init__.py
    ├── jabref_loader.py  # Jabref/BibTeX loader
    └── datatable_loader.py  # CSV/JSON/Excel loader
```

## Custom Metrics

You can select specific metrics to use:

```python
# Only use faithfulness and relevance
evaluator = RAGEvaluator(metrics=['faithfulness', 'relevance'])
results = evaluator.evaluate(query, context, answer)
```

## API Reference

### RAGEvaluator

Main class for evaluating RAG models.

**Methods**:
- `evaluate(query, context, answer, ground_truth=None)`: Evaluate a single output
- `evaluate_batch(queries, contexts, answers, ground_truths=None)`: Evaluate multiple outputs
- `get_average_scores(batch_results)`: Calculate average scores from batch results

### DataTableLoader

Loader for tabular data formats.

**Methods**:
- `load(file_path, format=None)`: Load data from file
- `load_for_evaluation(file_path, ...)`: Load data ready for evaluation

**Supported Formats**: CSV, JSON, Excel (.xlsx, .xls)

### JabrefLoader

Loader for Jabref/BibTeX format.

**Methods**:
- `load(file_path)`: Load data from BibTeX file
- `load_for_evaluation(file_path)`: Load data ready for evaluation

**Supported Formats**: BibTeX (.bib), JSON

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{rag_evaluation,
  title={RAG Evaluation Framework},
  author={RESHAPELab},
  year={2026},
  url={https://github.com/RESHAPELab/RAG-evaluation}
}
```

## Contact

For questions or issues, please open an issue on GitHub.