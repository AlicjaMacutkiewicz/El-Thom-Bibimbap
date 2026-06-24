# Persistence-Gated GRU-RK4 Model

## Motivation

The generalized synthetic ablation showed that GRU-RK4 with trajectory loss is
the strongest learned model, while the real-flight replay showed that simple
acceleration persistence is the stronger local predictor. The fourth ablation
therefore tests a specific hypothesis: the model should learn when the hybrid
physics forecast is useful and otherwise carry forward the last observed
acceleration.

This is a two-expert mixture:

- persistence expert: the last acceleration observed before telemetry cutoff;
- learned physics expert: conditioned RK4 acceleration plus the GRU residual.

For axis `j` and forecast step `t`, the model emits a residual and gate logit:

```text
a_hybrid[t,j] = a_RK4[t,j] + residual_GRU[t,j]
gate[t,j] = sigmoid(gate_logit[t,j])
a_pred[t,j] = a_last[j] + gate[t,j] * (a_hybrid[t,j] - a_last[j])
```

The gate is initialized with zero weights and bias `-2`, giving an initial value
of approximately `0.119`. Training therefore starts close to persistence; the
learned branch must demonstrate lower loss before receiving more weight.

## Objective

The model retains the acceleration and integrated-trajectory losses used by
`gru_res_phys`. It adds a baseline-relative trajectory regret term:

```text
L = L_acc + 0.2 * L_trajectory
    + lambda_regret * max(0, L_trajectory - L_persistence_trajectory)
```

The default `lambda_regret` is `0.1`. This term does not promise that the model
will dominate persistence outside the training distribution. It directly trains
the gate to avoid learned corrections in synthetic windows where they make the
trajectory worse than persistence.

Only the last pre-cutoff acceleration is used as the anchor. No future sample is
used by the model.

## Research Basis

- Jacobs, Jordan, Nowlan, and Hinton, *Adaptive Mixtures of Local Experts*,
  Neural Computation 3(1), 1991. A gating network learns input-dependent weights
  for specialized experts: https://www.cs.toronto.edu/~hinton/absps/jacobs.pdf
- Srivastava, Greff, and Schmidhuber, *Highway Networks*, 2015. Learned transform
  gates regulate a transformed branch against a carry path; negative transform
  gate bias favors carrying the baseline early in training:
  https://arxiv.org/abs/1505.00387
- Lim, Arik, Loeff, and Pfister, *Temporal Fusion Transformers for Interpretable
  Multi-horizon Time Series Forecasting*, International Journal of Forecasting,
  2021. Gating suppresses components that are unnecessary for a forecasting
  regime: https://arxiv.org/abs/1912.09363
- Jia et al., *Physics-Guided Recurrent Neural Networks for Modeling Dynamical
  Systems*, 2019. Physics-model outputs and recurrent learning are combined for
  hybrid dynamical forecasting: https://arxiv.org/abs/1810.02880

## Training

Use model type `gru_res_phys_persist_gate`. Train it from a fresh initialization;
its six-output checkpoint is intentionally incompatible with the three-output
heads of the existing ablation models.

```bash
python main.py \
  --output-dir ../../../../data \
  --batch-size 36000 \
  --training-rounds 200 \
  --seq-len 100 \
  --num-flights 1652 \
  --model-type gru_res_phys_persist_gate \
  --persistence-regret-weight 0.1
```

The best checkpoint is written as `best_gated_gru_model_seq100.pth`, so training
this variant does not overwrite `best_gru_model_seq100.pth` from an existing
three-output run.

The experiment is successful only if the fourth model is evaluated on the same
held-out synthetic flights and the same real-flight replay as the other three.
The gate should not be tuned directly on the real flight used for reporting.
