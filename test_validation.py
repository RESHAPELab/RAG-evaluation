"""
Test validation features of RagasEvaluator
"""
import sys
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).parent))

def test_invalid_metric_names():
    """Test that invalid metric names raise ValueError"""
    try:
        from rag_evaluation import RagasEvaluator
        
        # Set a fake API key for testing
        os.environ['OPENAI_API_KEY'] = 'test-key'
        
        try:
            evaluator = RagasEvaluator(metrics=['invalid_metric', 'faithfulness'])
            print("✗ Should have raised ValueError for invalid metric")
            return False
        except ValueError as e:
            if "Invalid metric names" in str(e) and "invalid_metric" in str(e):
                print("✓ Invalid metric names correctly rejected")
                return True
            else:
                print(f"✗ Unexpected error message: {e}")
                return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False
    finally:
        # Clean up
        if 'OPENAI_API_KEY' in os.environ and os.environ['OPENAI_API_KEY'] == 'test-key':
            del os.environ['OPENAI_API_KEY']

def test_batch_length_validation():
    """Test that batch evaluation validates input lengths"""
    try:
        from rag_evaluation import RagasEvaluator
        
        # Set a fake API key for testing
        os.environ['OPENAI_API_KEY'] = 'test-key'
        
        evaluator = RagasEvaluator()
        
        # Test with mismatched lengths
        try:
            results = evaluator.evaluate_batch(
                queries=["Q1", "Q2"],
                contexts=["C1"],  # Only 1 context for 2 queries
                answers=["A1", "A2"]
            )
            print("✗ Should have raised ValueError for mismatched lengths")
            return False
        except ValueError as e:
            if "same length" in str(e):
                print("✓ Mismatched input lengths correctly rejected")
                return True
            else:
                print(f"✗ Unexpected error message: {e}")
                return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False
    finally:
        # Clean up
        if 'OPENAI_API_KEY' in os.environ and os.environ['OPENAI_API_KEY'] == 'test-key':
            del os.environ['OPENAI_API_KEY']

if __name__ == "__main__":
    print("Running validation tests...\n")
    
    tests = [
        ("Invalid metric names", test_invalid_metric_names),
        ("Batch length validation", test_batch_length_validation),
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
        print("All validation tests passed! ✓")
        sys.exit(0)
    else:
        print(f"Some tests failed ({total - passed} failures)")
        sys.exit(1)
