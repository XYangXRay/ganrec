# GANrec: A GAN-based data reconstruction framework

## Overview

GANrec is an data reconstruction framework that harnesses the power of Generative Adversarial Networks (GANs). While traditional reconstruction methods primarily rely on intricate algorithms to piece together fragmented data, GANrec employs the generative capabilities of GANs to reimagine and revitalize data reconstruction.

Originally designed for the fields of tomography and phase retrieval, GANrec shines in its adaptability. With a provision to input the forward model, the framework can be flexibly adapted for complex data reconstruction processes across diverse applications.

## Features

1. GAN-powered Reconstruction: At its core, GANrec employs GANs to assist in the reconstruction process, enabling more accurate and efficient results than conventional methods.
2. Specialized for Tomography & Phase Retrieval: GANrec has been optimized for tomography and phase retrieval applications, ensuring precision and reliability in these domains.
3. Modular Design: The framework's architecture allows users to provide their forward model, making it adaptable for various complex data reconstruction challenges.
4. Efficient and Scalable: Built to handle large datasets, GANrec ensures that speed and efficiency are maintained without compromising the accuracy of reconstruction.

## Installation

The simplest way is pip. Pixi and Conda sections below show example runtime environments.

### Pip (quickest)

Install GANrec with your preferred backend in one step using extras:

```bash
# TensorFlow backend
pip install "ganrec[tensorflow]"

# PyTorch backend
pip install "ganrec[pytorch]"
```

Notes:
- For TensorFlow GPU wheels, see TensorFlow’s docs (e.g., tensorflow[and-cuda]) and ensure compatible NVIDIA drivers/CUDA.
- For PyTorch, use the selector on pytorch.org for OS/CUDA-specific commands if the generic wheel doesn’t match your system.

### Pixi (recommended)

Pixi manages environments and dependencies declaratively via `pyproject.toml`.

- Default backend (TensorFlow with CUDA extras; no PyTorch):

```bash
pixi install
```

- Optional PyTorch backend (no TensorFlow):

```bash
pixi -e pytorch install
```

Quick start with Pixi:

1) Install Pixi (see https://pixi.sh)
2) Clone this repo and enter it:
   - git clone https://github.com/XYangXRay/ganrec.git && cd ganrec
3) Install the default environment (TensorFlow):
   - pixi install
4) Or install the PyTorch environment only:
   - pixi -e pytorch install
5) Run examples:
   - pixi run python examples/holography_tf.py
   - pixi -e pytorch run python examples/tomography_torch.py

Note: Environments are defined in `pyproject.toml` — `default` uses TensorFlow; `pytorch` uses PyTorch.

### Conda environment (example runtime)

#### 1. Create & Activate a Conda Environment
Open your terminal or command prompt and create a new conda environment named `ganrec` with Python 3.11:

```bash
conda create --name ganrec python=3.11
conda activate ganrec
```

#### 2. Install ganrec with your preferred backend
Then install via pip extras as above.


## Additional Notes for Users

### Choosing Between TensorFlow and PyTorch
If you're not sure which one to choose, consider the specific requirements of your project or any existing familiarity you have with either library.
TensorFlow is often chosen for its production deployment capabilities and integration with TensorFlow Extended (TFX).
PyTorch is favored for its ease of use, dynamic computation graph, and strong support from the research community.
### Installing GPU Support
ganrec requires heavy duty work with GPU, make sure to install the GPU versions of TensorFlow or PyTorch. Instructions for this can be found on the respective official websites (TensorFlow and PyTorch).

## Examples

GANrec currently has applications for tomography reconstruction and in-line phase contrast (holography) phase retrieval:

1. X-ray tomography reconstruction:
   - [Tomography Example](https://github.com/XYangXRay/ganrec/blob/main/examples/tomography_tf.ipynb)
2. Holography phase retrieval:
   - [Phase retrieval Example](https://github.com/XYangXRay/ganrec/blob/main/examples/holography_tf.ipynb)

You can also run the Python examples directly:

```bash
# TensorFlow tomography example
pixi run python examples/tomography_tf.py

# TensorFlow holography example
pixi run python examples/holography_tf.py

# PyTorch tomography example
pixi -e pytorch run python examples/tomography_torch.py
```

## References

If you find GANrec is useful in your work, please consider citing:

J. Synchrotron Rad. (2020). 27, 486-493.
Available at: https://doi.org/10.1107/S1600577520000831
