# Mac Setup
This is the setup that worked for me on mac 2025-11-04. 


```bash
# Create env
conda create -n shortest-paths-nn python=3.10
conda activate shortest-paths-nn

# Install PyTorch 2.4.0 
uv pip install "torch==2.4.0" "torchvision" "torchaudio==2.4.0"

# Install PyTorch Geometric dependencies
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
echo "TORCH_VERSION: ${TORCH_VERSION}"
uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+cpu.html"

# Install PyTorch Geometric
uv pip install torch-geometric

# Other dependencies
uv pip install -r requirements.txt
```

After setup, you can run first-stage single-terrain training with:

`./run-experiment-1-terrain.sh experiment-configs/new-MLP-test/GNN+MLP-real.yml debug --new`
