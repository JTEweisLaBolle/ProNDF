"""
Test script for the noise floor feature in plot_true_pred function.

This script creates a simple synthetic dataset and a minimal mock model
to test the noise floor visualization functionality.

Usage:
    python test_noise_floor.py

This will generate 4 test plots showing different noise floor scenarios:
    1. No noise floor (baseline)
    2. Single noise variance for all sources
    3. Different noise variance per source
    4. Large noise variance (for visibility)

To use with your own trained model and dataset:
    from plotting import plot_true_pred
    
    # With single noise variance (same for all sources)
    fig = plot_true_pred(model, test_dataset, noise_variance=0.01)
    
    # With different noise per source
    fig = plot_true_pred(model, test_dataset, noise_variance=[0.01, 0.02, 0.005])
    
    # Without noise floor (default)
    fig = plot_true_pred(model, test_dataset)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from data import MultiFidelityDataset
from plotting import plot_true_pred


class MockModel:
    """
    Minimal mock model that mimics the ProNDF interface for testing plotting functions.
    """
    def __init__(self, dsource=2, dtargets=1):
        self.dsource = dsource
        self.dtargets = dtargets
        # Create a simple deterministic block that just returns a linear function
        # This is a minimal implementation just for testing
        self.B3 = type('obj', (object,), {
            'probabilistic_output': False
        })()
        self.hparams = type('obj', (object,), {
            'dsource': dsource,
            'dtargets': dtargets
        })()
    
    def eval(self):
        return self
    
    def to(self, device):
        return self
    
    def get_model_outputs(self, batch):
        """
        Simple mock that returns predictions close to targets (with some error).
        For testing purposes, we'll make predictions that are slightly off.
        """
        # Get targets from batch
        targets = batch['targets']
        # Add some prediction error (simulate model imperfection)
        # Predictions will be close but not perfect
        preds = targets + 0.1 * torch.randn_like(targets)
        return {
            "B3": {
                "out": preds
            }
        }


def create_test_dataset(n_samples_per_source=100, dsource=2, noise_std=0.2):
    """
    Create a simple synthetic test dataset with known noise variance.
    
    Args:
        n_samples_per_source: Number of samples per source
        dsource: Number of sources
        noise_std: Standard deviation of noise to add to targets
    
    Returns:
        MultiFidelityDataset object
    """
    # Generate simple synthetic data
    source_list = []
    num_list = []
    targets_list = []
    
    for source_idx in range(dsource):
        # Create one-hot encoded source vectors
        source_oh = np.zeros((n_samples_per_source, dsource))
        source_oh[:, source_idx] = 1
        
        # Generate numerical inputs (1D for simplicity)
        x = np.linspace(0, 10, n_samples_per_source).reshape(-1, 1)
        
        # Generate targets: simple function + noise
        # Different function for each source
        if source_idx == 0:
            y_true = 2 * x.flatten() + 5
        else:
            y_true = -x.flatten() + 15
        
        # Add noise
        y_noisy = y_true + np.random.normal(0, noise_std, size=y_true.shape)
        
        source_list.append(source_oh)
        num_list.append(x)
        targets_list.append(y_noisy.reshape(-1, 1))
    
    # Concatenate all sources
    source_data = np.concatenate(source_list, axis=0)
    num_data = np.concatenate(num_list, axis=0)
    targets_data = np.concatenate(targets_list, axis=0)
    
    # Create metadata
    meta = {
        'dsource': dsource,
        'dcat': [],
        'dnum': 1,
        'dtargets': 1,
        'qual_in': False,
        'quant_in': True,
        'num_samples': [n_samples_per_source] * dsource
    }
    
    return MultiFidelityDataset(
        source=source_data,
        cat=None,
        num=num_data,
        targets=targets_data,
        meta=meta
    )


def test_noise_floor_plotting():
    """
    Test the noise floor plotting functionality with different scenarios.
    """
    print("Creating test dataset...")
    # Create dataset with known noise variance
    noise_std = 0.2
    noise_variance = noise_std ** 2  # 0.04
    test_dataset = create_test_dataset(
        n_samples_per_source=100,
        dsource=2,
        noise_std=noise_std
    )
    
    print("Creating mock model...")
    model = MockModel(dsource=2, dtargets=1)
    
    print("\nTesting plot_true_pred with different noise_variance options:\n")
    
    # Test 1: No noise floor (default behavior)
    print("1. Testing without noise floor (noise_variance=None)...")
    fig1 = plot_true_pred(
        model, 
        test_dataset, 
        device='cpu',
        noise_variance=None
    )
    fig1.suptitle("Test 1: No Noise Floor", fontsize=14, y=1.02)
    plt.savefig("test_noise_floor_1_none.png", dpi=150, bbox_inches='tight')
    print("   Saved: test_noise_floor_1_none.png")
    plt.close()
    
    # Test 2: Single noise variance for all sources
    print("2. Testing with single noise variance (same for all sources)...")
    fig2 = plot_true_pred(
        model, 
        test_dataset, 
        device='cpu',
        noise_variance=noise_variance
    )
    fig2.suptitle(f"Test 2: Single Noise Variance (σ²={noise_variance:.3f})", fontsize=14, y=1.02)
    plt.savefig("test_noise_floor_2_single.png", dpi=150, bbox_inches='tight')
    print(f"   Saved: test_noise_floor_2_single.png (noise_variance={noise_variance:.3f})")
    plt.close()
    
    # Test 3: Different noise variance per source
    print("3. Testing with different noise variance per source...")
    noise_variances = [noise_variance, noise_variance * 1.5]  # Different for each source
    fig3 = plot_true_pred(
        model, 
        test_dataset, 
        device='cpu',
        noise_variance=noise_variances
    )
    fig3.suptitle(
        f"Test 3: Per-Source Noise Variance (σ²={noise_variances[0]:.3f}, {noise_variances[1]:.3f})", 
        fontsize=14, 
        y=1.02
    )
    plt.savefig("test_noise_floor_3_per_source.png", dpi=150, bbox_inches='tight')
    print(f"   Saved: test_noise_floor_3_per_source.png (noise_variance={noise_variances})")
    plt.close()
    
    # Test 4: Wrong noise variance (to show the lines are visible)
    print("4. Testing with larger noise variance (to show visibility)...")
    large_noise_variance = noise_variance * 4  # Much larger
    fig4 = plot_true_pred(
        model, 
        test_dataset, 
        device='cpu',
        noise_variance=large_noise_variance
    )
    fig4.suptitle(f"Test 4: Large Noise Variance (σ²={large_noise_variance:.3f})", fontsize=14, y=1.02)
    plt.savefig("test_noise_floor_4_large.png", dpi=150, bbox_inches='tight')
    print(f"   Saved: test_noise_floor_4_large.png (noise_variance={large_noise_variance:.3f})")
    plt.close()
    
    print("\nAll tests completed! Check the generated PNG files to see the noise floor lines.")
    print("\nExpected behavior:")
    print("  - Red dashed lines should appear at y = x ± sqrt(noise_variance)")
    print("  - These lines indicate the noise floor (1σ bounds)")
    print("  - Points within these bounds are within the expected noise level")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run tests
    test_noise_floor_plotting()

