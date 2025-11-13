## Rational Policy Gradient

![RPG explainer](https://rational-policy-gradient.github.io/assets/figures/rpg-fig1.png)

This repository implements the Rational Policy Gradient (RPG) algorithms and supporting environments used in the 2025 NeurIPS paper. It provides experiment code, environment wrappers, utilities, and a small interactive demo to get started.

View our website here: https://rational-policy-gradient.github.io.

You can also experiment with the RPG package at this demo hosted on Colab: https://colab.research.google.com/drive/1uQnIK9aoYffOQpppfMwcIcjSPnOHoh2x?copy=true.

## Quick start — demo

The recommended starting point is the interactive demo notebook: `demo.ipynb` (identical to the Colab).
Open it to see examples that build RPG algorithms using the core API and shows how to run training and evaluation pipelines.

## Installation

Clone the repo and install the package and development extras.

```bash
git clone https://github.com/niklaslauffer/rational-policy-gradient
cd rational-policy-gradient
conda create -n rpg python=3.11 -y
conda activate rpg
pip install -e .[dev,gpu]
# Optional: install JAX with CUDA support if you will run on GPU
# pip install -U "jax[cuda12]"
```

### Verify installation

Run a quick test to verify the core script runs:

```bash
python rational_policy_gradient/rpg.py --config-name=rpg_matrix
```

## Usage

- Experiment configurations are stored in `data/configs/` (YAML files). Use the `--config-name` flag to select one.
- The primary program entrypoint is `rational_policy_gradient/rpg.py` which reads configs and runs training/evaluation.
- `rational_policy_gradient/rpg_algs.py` contains definitions for several RPG algorithms and can be used as examples for building new ones. The "RPG_ALG" entry in the YAML configs can be used to specify any of the preexisting RPG algorithms in `rpg_algs.py`.

## Citation

If you use this code in your research, please cite:

```
@inproceedings{lauffer2025rpg,
  title={Robust and Diverse Multi-Agent Learning via Rational Policy Gradient},
  author={Lauffer, Niklas and Shah, Ameesh and Carroll, Micah and Seshia, Sanjit A and Russell, Stuart and Dennis, Michael D},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```