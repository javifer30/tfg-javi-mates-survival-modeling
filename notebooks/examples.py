"""
Example script demonstrating how to use the refactored modules.
"""
import numpy as np
from src.data.data_loader import load_data
from src.features.pwe_transformer import PWEDataExpander
from src.models.deepsurv import DeepSurvNet, concordance_index
from src.models.pwe_poisson import PWEPoisson, PWEDataset
import torch


def example_data_loading():
    """Example: Load data using data_loader module."""
    print("=" * 50)
    print("Example 1: Data Loading")
    print("=" * 50)
    
    # Load processed data
    df = load_data('data/processed/icu_master.csv')
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()[:5]}...")
    print()


def example_pwe_transformation():
    """Example: Transform data to PWE format."""
    print("=" * 50)
    print("Example 2: PWE Transformation")
    print("=" * 50)
    
    # Simulate some data
    n_samples = 100
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    time = np.random.uniform(0, 10, n_samples)
    event = np.random.binomial(1, 0.5, n_samples)
    
    # Define time intervals (0-1, 1-2, ..., 9-10 days)
    breaks = np.arange(0, 11, 1)
    
    # Create and apply transformer
    pwe_expander = PWEDataExpander(breaks=breaks)
    df_pwe = pwe_expander.transform(X, time, event)
    
    print(f"Original data: {n_samples} patients")
    print(f"PWE expanded: {len(df_pwe)} rows")
    print(f"Columns: {df_pwe.columns.tolist()}")
    print(f"\nFirst 5 rows for patient 0:")
    print(df_pwe[df_pwe['id'] == 0].head())
    print()


def example_deepsurv_model():
    """Example: Create and use DeepSurv model."""
    print("=" * 50)
    print("Example 3: DeepSurv Model")
    print("=" * 50)
    
    # Model parameters
    n_features = 20
    hidden_layers = [64, 32]
    
    # Create model
    model = DeepSurvNet(in_features=n_features, hidden=hidden_layers)
    print(f"Model architecture:\n{model}")
    
    # Simulate prediction
    X_test = torch.randn(10, n_features)
    with torch.no_grad():
        risk_scores = model(X_test)
    
    print(f"\nRisk scores shape: {risk_scores.shape}")
    print(f"Sample risk scores: {risk_scores[:3].numpy()}")
    print()


def example_concordance_index():
    """Example: Calculate C-index."""
    print("=" * 50)
    print("Example 4: Concordance Index")
    print("=" * 50)
    
    # Simulate survival data
    time = np.array([5.0, 3.0, 8.0, 2.0, 6.0])
    event = np.array([1, 1, 0, 1, 1])
    risk_scores = np.array([0.8, 0.9, 0.3, 1.0, 0.5])
    
    c_index = concordance_index(time, event, risk_scores)
    print(f"Time: {time}")
    print(f"Event: {event}")
    print(f"Risk scores: {risk_scores}")
    print(f"C-index: {c_index:.4f}")
    print()


def example_pwe_model():
    """Example: Create PWE Poisson model."""
    print("=" * 50)
    print("Example 5: PWE Poisson Model")
    print("=" * 50)
    
    # Simulate PWE data
    n_samples = 100
    n_features = 10
    X = np.random.randn(n_samples, n_features)
    time = np.random.uniform(0, 10, n_samples)
    event = np.random.binomial(1, 0.5, n_samples)
    
    # Transform to PWE format
    breaks = np.arange(0, 11, 1)
    pwe_expander = PWEDataExpander(breaks=breaks)
    df_pwe = pwe_expander.transform(X, time, event)
    
    # Create PWE dataset
    pwe_dataset = PWEDataset(df_pwe, p=n_features)
    print(f"PWE dataset length: {len(pwe_dataset)}")
    
    # Create model
    K = len(breaks) - 1  # number of intervals
    model = PWEPoisson(p=n_features, K=K)
    print(f"PWE model: {n_features} features, {K} intervals")
    
    # Sample forward pass
    X_batch, k_batch, log_y_batch, d_batch, id_batch = pwe_dataset[0]
    X_batch = X_batch.unsqueeze(0)
    k_batch = k_batch.unsqueeze(0)
    log_y_batch = log_y_batch.unsqueeze(0)
    
    with torch.no_grad():
        log_mu = model(X_batch, k_batch, log_y_batch)
    
    print(f"Sample log(μ): {log_mu.item():.4f}")
    print()


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 48 + "╗")
    print("║" + " " * 10 + "REFACTORED MODULE EXAMPLES" + " " * 11 + "║")
    print("╚" + "=" * 48 + "╝")
    print()
    
    # Run examples (comment out data loading if file doesn't exist)
    # example_data_loading()
    example_pwe_transformation()
    example_deepsurv_model()
    example_concordance_index()
    example_pwe_model()
    
    print("=" * 50)
    print("All examples completed successfully!")
    print("=" * 50)
