#!/usr/bin/env python
"""
Quick validation script for EmbeddingGenerator.

Tests basic functionality without requiring API calls.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from langchain_core.documents import Document
from src.marketing_agents.rag.embedding_generator import EmbeddingGenerator


def test_initialization():
    """Test that EmbeddingGenerator can be initialized."""
    print("Testing initialization...")

    try:
        embedder = EmbeddingGenerator(
            model_name="text-embedding-3-small", batch_size=100, enable_cache=True
        )
        print("✓ Initialization successful")
        print(f"  Model: {embedder.model_name}")
        print(f"  Batch size: {embedder.batch_size}")
        print(f"  Expected dimension: {embedder.expected_dimension}")
        print(f"  Cache enabled: {embedder.enable_cache}")
        return True
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False


def test_model_dimensions():
    """Test model dimension mapping."""
    print("\nTesting model dimensions...")

    models = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    all_passed = True
    for model, expected_dim in models.items():
        embedder = EmbeddingGenerator(model_name=model)
        if embedder.expected_dimension == expected_dim:
            print(f"✓ {model}: {expected_dim} dimensions")
        else:
            print(
                f"✗ {model}: Expected {expected_dim}, got {embedder.expected_dimension}"
            )
            all_passed = False

    return all_passed


def test_content_hashing():
    """Test content hash generation."""
    print("\nTesting content hashing...")

    try:
        embedder = EmbeddingGenerator()

        # Same content should produce same hash
        text1 = "This is a test"
        text2 = "This is a test"
        text3 = "Different text"

        hash1 = embedder._get_content_hash(text1)
        hash2 = embedder._get_content_hash(text2)
        hash3 = embedder._get_content_hash(text3)

        assert hash1 == hash2, "Same content should produce same hash"
        assert hash1 != hash3, "Different content should produce different hash"
        assert len(hash1) == 64, "SHA256 hash should be 64 hex characters"

        print("✓ Content hashing works correctly")
        return True
    except Exception as e:
        print(f"✗ Content hashing failed: {e}")
        return False


def test_validation_structure():
    """Test validation method structure (without embeddings)."""
    print("\nTesting validation structure...")

    try:
        embedder = EmbeddingGenerator()

        # Create mock valid embeddings
        documents = [
            Document(page_content="Test doc 1", metadata={"source": "test1.txt"}),
            Document(page_content="Test doc 2", metadata={"source": "test2.txt"}),
        ]

        embeddings = [(doc, [0.1] * 1536) for doc in documents]

        validation = embedder.validate_embeddings(embeddings)

        # Check validation result structure
        required_keys = [
            "total_embeddings",
            "expected_dimension",
            "dimension_valid",
            "contains_nan",
            "contains_inf",
            "validation_passed",
        ]

        for key in required_keys:
            assert key in validation, f"Missing key: {key}"

        assert validation["total_embeddings"] == 2
        assert validation["expected_dimension"] == 1536
        assert validation["validation_passed"] is True

        print("✓ Validation structure correct")
        print(f"  Total embeddings: {validation['total_embeddings']}")
        print(f"  Dimension: {validation['expected_dimension']}")
        print(f"  Validation passed: {validation['validation_passed']}")
        return True
    except Exception as e:
        print(f"✗ Validation structure test failed: {e}")
        return False


def test_stats_structure():
    """Test statistics structure."""
    print("\nTesting statistics structure...")

    try:
        embedder = EmbeddingGenerator()
        stats = embedder.get_stats()

        required_keys = [
            "total_processed",
            "cache_hits",
            "cache_misses",
            "failed_embeddings",
            "retries",
            "cache_hit_rate",
            "model",
            "batch_size",
        ]

        for key in required_keys:
            assert key in stats, f"Missing key: {key}"

        print("✓ Statistics structure correct")
        print(f"  Model: {stats['model']}")
        print(f"  Batch size: {stats['batch_size']}")
        print(f"  Cache hit rate: {stats['cache_hit_rate']}")
        return True
    except Exception as e:
        print(f"✗ Statistics structure test failed: {e}")
        return False


def test_dimension_validation():
    """Test dimension mismatch detection."""
    print("\nTesting dimension validation...")

    try:
        embedder = EmbeddingGenerator()

        documents = [
            Document(page_content="Test doc 1", metadata={"source": "test1.txt"}),
            Document(page_content="Test doc 2", metadata={"source": "test2.txt"}),
        ]

        # One correct, one incorrect dimension
        embeddings = [
            (documents[0], [0.1] * 1536),  # Correct
            (documents[1], [0.1] * 512),  # Wrong dimension
        ]

        validation = embedder.validate_embeddings(embeddings)

        assert validation["validation_passed"] is False, "Should fail validation"
        assert (
            validation["dimension_valid"] is False
        ), "Should detect dimension mismatch"
        assert len(validation["dimension_mismatches"]) == 1, "Should have 1 mismatch"

        print("✓ Dimension validation works correctly")
        print(f"  Detected {len(validation['dimension_mismatches'])} mismatch(es)")
        return True
    except Exception as e:
        print(f"✗ Dimension validation failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("EmbeddingGenerator Validation Tests")
    print("=" * 60)

    tests = [
        test_initialization,
        test_model_dimensions,
        test_content_hashing,
        test_validation_structure,
        test_stats_structure,
        test_dimension_validation,
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test {test_func.__name__} crashed: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("✓ All validation tests passed!")
        return 0
    else:
        print(f"✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
