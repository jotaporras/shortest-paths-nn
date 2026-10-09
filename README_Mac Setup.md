# Mac Setup
This setup is for the `asilomar` branch (Apple silicon, CPU only).
An older setup from 2025-11-04 used torch 2.4.0 and the env `shortest-paths-nn`.

```bash
# Create env
conda create -n spnn-asilomar python=3.11
conda activate spnn-asilomar

# Install PyTorch
uv pip install "torch==2.5.1"

# Install PyTorch Geometric dependencies
uv pip install torch_scatter torch_sparse -f "https://data.pyg.org/whl/torch-2.5.1+cpu.html"

# Install PyTorch Geometric and the other dependencies
uv pip install "torch-geometric==2.6.1" -r requirements.txt
uv pip install -e ".[dev]"
```

Check the setup with the test suite. The SparseGT tests need MKL, so they skip on Mac.

`pytest tests -q`

After setup, you can run single-stage training with `python -m spnn.cli.train`.
See the "Asilomar branch: quick start" section of `README.md` for the flags.
