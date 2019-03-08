# image-registration-sinkhorn-divergence

This repository provides efficient implementations of
Maximum Mean Discrepancies (a.k.a. kernel norms),
Hausdorff and Sinkhorn divergences between sampled measures motivated from [Feydy et al., 2018](https://arxiv.org/abs/1810.08278).
Thanks to the [KeOps library](https://www.kernel-operations.io),
our routines scale up to batches of 1,000,000 samples, **without memory overflows**.


First and foremost, this repo is about providing a reference implementation of Sinkhorn-related divergences. In [`/common/`](./common), you will find
a [simple](./common/sinkhorn_balanced_simple.py) and
an [efficient](./common/sinkhorn_balanced.py) implementation of
the Sinkhorn algorithm.


One can follow the contents of the [code_walkthrough](./code_walkthrough.ipynb) to get a rough idea on an abstract level that gives the similarity score between source and target images. While the [score_from_training_phase.py](./score_from_training_phase.py) normalizes the overall score to {0,1,2,3} of an individual by keeping track of the previous scores in training phase, where 0 means normal range and 3 means that the handwriting is deteriorating. Hence, the latter is use-case specific so one is referred to only browse the jupyter notebook.

Please note that for smooth running of our code, you will need to install
both [pytorch 0.4.1](https://pytorch.org/) and [KeOps](https://www.kernel-operations.io). No worries: `pip install pykeops` should do the trick.

