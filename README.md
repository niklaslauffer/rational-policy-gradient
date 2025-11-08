# Installation

```
git clone https://github.com/niklaslauffer/rational-policy-gradient
conda create -n rppo python=3.11.11 -y
conda activate rppo
cd rational-policy-gradient
pip install -e .[gpu,dev]
(optional to run on GPU) pip install -U "jax[cuda12]"
```

### Verify installation

Simple command to verify that code runs:
```
python rational_policy_gradient/rpg.py --config-name=rpg_overcooked
```