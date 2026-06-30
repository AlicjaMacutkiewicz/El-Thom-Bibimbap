# Evaluation Deprecation Notice

The evaluation procedure previously implemented in this directory must not be
used for model-performance reporting or comparison with the GRU + RK4 model.

The previous evaluation is invalid for comparative claims because it:
- reconstructs normalization statistics from evaluation data;
- evaluates only a split/subset of the available evaluation corpus;
- assumes a fixed time step inconsistent with the evaluated data;
- initializes predictions using future-derived state information.

The `basic_fit` model may still be retained as a baseline implementation.
Any reported baseline metrics must be recomputed using the corrected external
evaluation protocol on the independent test dataset.