"""
Test script to verify Mamba, Transformer, and HMT models work correctly.
Run: python test_models.py
"""
import torch
import torch.nn as nn

def test_mamba():
    print("\n=== Testing MambaBlock ===")
    from implementations import MambaBlock
    
    batch, seq, dim = 4, 32, 64
    x = torch.randn(batch, seq, dim, requires_grad=True)
    
    model = MambaBlock(dim, d_state=16, expand=2)
    model.train()
    
    out = model(x)
    
    print(f"  Input:  {x.shape} -> Output: {out.shape}")
    assert out.shape == x.shape, "Shape mismatch!"
    
    loss = out.mean()
    loss.backward()
    print(f"  Forward OK, backward OK (loss={loss.item():.4f})")
    return True

def test_transformer():
    print("\n=== Testing TransformerBlock ===")
    from implementations import TransformerBlock
    
    batch, seq, dim = 4, 32, 64
    x = torch.randn(batch, seq, dim, requires_grad=True)
    
    model = TransformerBlock(dim, num_heads=4, ff_mult=4, max_len=128)
    model.train()
    
    out = model(x)
    
    print(f"  Input:  {x.shape} -> Output: {out.shape}")
    assert out.shape == x.shape, "Shape mismatch!"
    
    loss = out.mean()
    loss.backward()
    print(f"  Forward OK, backward OK (loss={loss.item():.4f})")
    return True

def test_hmt():
    print("\n=== Testing HierarchicalMambaTransformer ===")
    from implementations import HierarchicalMambaTransformer
    
    batch, seq, dim = 4, 32, 128
    x = torch.randn(batch, seq, dim, requires_grad=True)
    
    model = HierarchicalMambaTransformer(
        dim=dim, num_layers=2, num_heads=4, num_scales=2, max_seq=128
    )
    model.train()
    
    out = model(x)
    
    print(f"  Input:  {x.shape} -> Output: {out.shape}")
    assert out.shape == x.shape, "Shape mismatch!"
    
    loss = out.mean()
    loss.backward()
    print(f"  Forward OK, backward OK (loss={loss.item():.4f})")
    return True

def test_classifier():
    print("\n=== Testing HMTClassifier ===")
    from implementations import HMTClassifier
    
    batch, feat_dim = 8, 16
    x = torch.randn(batch, feat_dim, requires_grad=True)
    
    model = HMTClassifier(
        input_dim=feat_dim, num_classes=1,
        dim=64, num_layers=2, num_heads=4, num_scales=2
    )
    model.train()
    
    out = model(x)
    
    print(f"  Input:  {x.shape} -> Output: {out.shape}")
    assert out.shape == (batch, 1), f"Expected ({batch}, 1), got {out.shape}"
    
    loss = out.mean()
    loss.backward()
    print(f"  Forward OK, backward OK (loss={loss.item():.4f})")
    return True

def test_language_model():
    print("\n=== Testing HMTLanguageModel ===")
    from implementations import HMTLanguageModel
    
    batch, seq, vocab = 4, 32, 256
    x = torch.randint(0, vocab, (batch, seq))
    
    model = HMTLanguageModel(
        vocab_size=vocab, dim=64, num_layers=2,
        num_heads=4, num_scales=2, max_seq=seq
    )
    model.train()
    
    out = model(x)  # (B, T, vocab)
    
    print(f"  Input:  {x.shape} -> Output: {out.shape}")
    assert out.shape == (batch, seq, vocab), f"Expected ({batch}, {seq}, {vocab})"
    
    loss = nn.functional.cross_entropy(
        out.view(-1, vocab), x.view(-1), ignore_index=0
    )
    loss.backward()
    print(f"  Forward OK, backward OK (loss={loss.item():.4f})")
    return True

def main():
    print("=" * 50)
    print("MODEL ARCHITECTURE TESTS")
    print("=" * 50)
    
    results = []
    
    try:
        results.append(("MambaBlock", test_mamba()))
    except Exception as e:
        print(f"  FAILED: {e}")
        results.append(("MambaBlock", False))
    
    try:
        results.append(("TransformerBlock", test_transformer()))
    except Exception as e:
        print(f"  FAILED: {e}")
        results.append(("TransformerBlock", False))
    
    try:
        results.append(("HierarchicalMambaTransformer", test_hmt()))
    except Exception as e:
        print(f"  FAILED: {e}")
        results.append(("HierarchicalMambaTransformer", False))
    
    try:
        results.append(("HMTClassifier", test_classifier()))
    except Exception as e:
        print(f"  FAILED: {e}")
        results.append(("HMTClassifier", False))
    
    try:
        results.append(("HMTLanguageModel", test_language_model()))
    except Exception as e:
        print(f"  FAILED: {e}")
        results.append(("HMTLanguageModel", False))
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")
    
    all_pass = all(ok for _, ok in results)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")

if __name__ == "__main__":
    main()
