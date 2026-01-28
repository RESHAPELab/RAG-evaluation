"""
Simple test to verify ragas integration
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def test_import():
    """Test that RagasEvaluator can be imported"""
    try:
        from rag_evaluation import RagasEvaluator
        print("✓ RagasEvaluator imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import RagasEvaluator: {e}")
        return False

def test_basic_evaluator():
    """Test that basic evaluator still works"""
    try:
        from rag_evaluation import RAGEvaluator
        
        evaluator = RAGEvaluator()
        results = evaluator.evaluate(
            query="What is machine learning?",
            context="Machine learning is a subset of AI.",
            answer="ML is a type of AI."
        )
        
        assert 'faithfulness' in results
        assert 'relevance' in results
        print("✓ Basic evaluator works correctly")
        return True
    except Exception as e:
        print(f"✗ Basic evaluator failed: {e}")
        return False

def test_ragas_initialization():
    """Test that RagasEvaluator can be initialized (without API key)"""
    try:
        from rag_evaluation import RagasEvaluator
        import os
        
        # This should fail without API key
        if not os.environ.get("OPENAI_API_KEY"):
            try:
                evaluator = RagasEvaluator()
                print("✗ RagasEvaluator should require OPENAI_API_KEY")
                return False
            except ValueError as e:
                if "OPENAI_API_KEY" in str(e):
                    print("✓ RagasEvaluator correctly requires OPENAI_API_KEY")
                    return True
                else:
                    print(f"✗ Unexpected error: {e}")
                    return False
        else:
            evaluator = RagasEvaluator()
            print("✓ RagasEvaluator initialized successfully with API key")
            return True
    except Exception as e:
        print(f"✗ RagasEvaluator initialization test failed: {e}")
        return False

if __name__ == "__main__":
    print("Running integration tests...\n")
    
    tests = [
        ("Import test", test_import),
        ("Basic evaluator test", test_basic_evaluator),
        ("Ragas initialization test", test_ragas_initialization),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{name}:")
        results.append(test_func())
    
    print("\n" + "="*50)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("All tests passed! ✓")
        sys.exit(0)
    else:
        print(f"Some tests failed ({total - passed} failures)")
        sys.exit(1)
