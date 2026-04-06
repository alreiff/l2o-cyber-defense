# l2o-cyber-defense
This repository contains research code associated with the ANR JCJC project on learning-based solvers for cybersecurity defense problems with graph structure and bi-level decision-making.

For related work using the Active Directory (AD) simulator in this setting, see the following repository:
[https://github.com/Lilianltd/Markov_Budget
](https://github.com/Lilianltd/Markov_Budget/tree/main/adsimulator_graph_generator)

This latter repository was developed by some of my students (Lilian LETARD, Jean VAN DYK, Louis VIRBEL) that I got the chance to work with, and we are currently working on merging it with the present codebase to provide a unified framework.
The codebase is organized as a **small library**:
- reusable code lives in `src/l2o_cyber_defense/`
- exploratory and paper-oriented experiments live in `notebooks/`
- command-line runs live in `scripts/`
- experiment presets live in `configs/`
- sanity checks live in `tests/`

## Repository structure

```text
l2o-cyber-defense/
├── src/l2o_cyber_defense/
├── notebooks/
│   ├── 00_slowing_down_original.ipynb
│   ├── 01_graph_generation_and_masks.ipynb
│   ├── 02_fixed_point_policy_experiments.ipynb
│   └── 03_training_theta_on_dag_dataset.ipynb
├── scripts/
│   ├── make_dag_dataset.py
│   ├── single_instance_demo.py
│   ├── smoke_demo.py
│   └── train_theta.py
├── configs/
│   ├── quickstart_train.json
│   ├── default_train.json
│   └── dataset_small.json
├── tests/
├── data/
├── figures/
└── results/
```

## Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/<your-username>/l2o-cyber-defense.git
cd l2o-cyber-defense
python -m venv .venv
source .venv/bin/activate
```

Install dependencies and the package in editable mode:

```bash
pip install -r requirements.txt
pip install -e .
```

## First steps

Run the smoke test:

```bash
python scripts/smoke_demo.py
```

Open the first notebook:

```bash
jupyter notebook notebooks/01_graph_generation_and_masks.ipynb
```

## How to test the code

Run all tests:

```bash
pytest -q
```

Run one test file:

```bash
pytest -q tests/test_fixed_point.py
```

Run one test function:

```bash
pytest -q tests/test_projections.py::test_proj_simplex_masked
```

A good habit is to run `pytest -q` each time you modify functions in `src/`.

## How to run experiments

### 1. Generate a dataset

```bash
python scripts/make_dag_dataset.py --config configs/dataset_small.json
```

This saves a `.pt` file with the matrices and a `.json` file with metadata.

### 2. Run a single-instance demo

```bash
python scripts/single_instance_demo.py
```

This is useful to check that the fixed-point policy solver behaves sensibly on one DAG instance.

### 3. Launch a quick training run

```bash
python scripts/train_theta.py --config configs/quickstart_train.json
```

This runs a short experiment and writes results to `results/quickstart_train.json`.

### 4. Launch a longer default run

```bash
python scripts/train_theta.py --config configs/default_train.json
```

### 5. Override one parameter from the command line

The config file gives defaults, but any CLI flag overrides it. For example:

```bash
python scripts/train_theta.py --config configs/default_train.json --epochs 20 --lr 0.01
```

## Suggested workflow for experiments

A simple workflow is:

1. Prototype an idea in a notebook.
2. Move reusable functions into `src/l2o_cyber_defense/`.
3. Add a small test in `tests/`.
4. Create a config file in `configs/`.
5. Run the experiment from `scripts/` and save outputs in `results/`.

This keeps the repo clean and makes figures easier to reproduce for papers and ANR reports.

## What to compare in your first experiments

A good first set of experiments is:

- vary `K` and `Ksum`
- vary the graph size range `nmin`, `nmax`
- vary `edge_prob`
- compare performance across random seeds
- compare learned `theta` against simple baselines such as uniform positive theta

For example, you can run:

```bash
python scripts/train_theta.py --config configs/quickstart_train.json --seed 0
python scripts/train_theta.py --config configs/quickstart_train.json --seed 1
python scripts/train_theta.py --config configs/quickstart_train.json --seed 2
```

and then compare the saved JSON files in `results/`.

## Citation

If you use this repository, please cite the associated ANR project and related papers when available.

## License

You can add an MIT license or the license required by your project/institution.
