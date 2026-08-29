"""The bundled demo scenario.

These constants define the claim LabGuard is shipped to validate.  The
weaknesses are real properties of the configuration, not scripted outputs: the
imbalance, the unequal training budget, the test-set checkpoint selection and
the divergent learning rate all genuinely change what the training loop
produces.  Every number the dashboard shows is computed at run time.
"""

from __future__ import annotations

from ..models.domain import (
    BudgetPolicy,
    Claim,
    ClaimContext,
    DatasetInfo,
    ExistingResult,
    ModelConfig,
)
from ..models.enums import AutonomyMode

DEMO_CLAIM_TEXT = "Model B performs better than Model A on the violence-detection benchmark."

#: The seed the original result was reported on. It is the most favourable of
#: the seeds tried, which is exactly what the seed-sensitivity check exposes.
ORIGINAL_SEED = 11

#: Seeds used by the verification sweep: the original plus a fixed, declared
#: series, so the choice of comparison seeds cannot itself be cherry-picked.
VERIFICATION_SEEDS = [ORIGINAL_SEED, 1, 2, 3, 4]

DEMO_DATASET = DatasetInfo(
    name="synthetic_violence_clips",
    n_samples=4000,
    n_features=24,
    positive_rate=0.08,
    test_fraction=0.25,
)

MODEL_A = ModelConfig(
    name="Model A",
    family="linear",
    hidden_units=0,
    epochs=25,
    learning_rate=0.10,
    batch_size=64,
    class_weight="sqrt_balanced",
    is_baseline=True,
    notes="Class-weighted linear baseline, trained for 25 epochs.",
)

MODEL_B = ModelConfig(
    name="Model B",
    family="mlp",
    hidden_units=24,
    epochs=90,
    learning_rate=0.20,
    batch_size=64,
    class_weight="none",
    is_baseline=False,
    notes="Higher-capacity MLP, trained for 90 epochs with no class weighting.",
)

#: A faster-converging setup the researcher also tried. Its divergence is
#: real, not injected: squared error on the raw logit provably diverges above
#: a critical learning rate, and 2.4 is above it for this data.
UNSTABLE_LR = 2.4
UNSTABLE_SEED = 3

MODEL_B_VARIANT = ModelConfig(
    name="Model B (fast-LR variant)",
    family="mlp",
    hidden_units=24,
    epochs=40,
    learning_rate=UNSTABLE_LR,
    batch_size=64,
    class_weight="none",
    objective="mse_logit",
    role="variant",
    notes="Squared-error head at learning rate 2.4, submitted as a faster setup.",
)

#: The run whose curve overfits, used by the early-stopping demonstration.
OVERFIT_SEED = 4

DEMO_EXISTING_RESULTS = [
    ExistingResult(
        model_name="Model A",
        metric="accuracy",
        value=0.891,
        seed=ORIGINAL_SEED,
        checkpoint_selected_on="test",
        epochs_trained=25,
    ),
    ExistingResult(
        model_name="Model B",
        metric="accuracy",
        value=0.912,
        seed=ORIGINAL_SEED,
        checkpoint_selected_on="test",
        epochs_trained=90,
        checkpoint_uri="gs://labguard-demo/checkpoints/model_b_seed11_epoch90.ckpt",
    ),
]


def demo_context() -> ClaimContext:
    return ClaimContext(
        dataset=DEMO_DATASET.model_copy(deep=True),
        models=[
            MODEL_A.model_copy(deep=True),
            MODEL_B.model_copy(deep=True),
            MODEL_B_VARIANT.model_copy(deep=True),
        ],
        existing_results=[r.model_copy(deep=True) for r in DEMO_EXISTING_RESULTS],
        reported_checkpoint_corrupt=True,
        notes=(
            "Reported on a single seed. Model B was trained for 90 epochs and "
            "Model A for 25. Checkpoints were selected on the test split."
        ),
    )


def demo_claim(autonomy: AutonomyMode = AutonomyMode.MANAGED_AUTONOMY) -> Claim:
    return Claim(
        text=DEMO_CLAIM_TEXT,
        context=demo_context(),
        autonomy_mode=autonomy,
        budget=BudgetPolicy(total_units=40.0, approval_threshold_units=6.0),
        demo_mode=True,
    )


def arms(context: ClaimContext) -> tuple[ModelConfig, ModelConfig]:
    """Return `(baseline, candidate)` from a claim context, with fallbacks."""
    models = [m for m in context.models if m.role == "primary"] or [MODEL_A, MODEL_B]
    baseline = next((m for m in models if m.is_baseline), models[0])
    candidate = next((m for m in models if m is not baseline), models[-1])
    return baseline, candidate


def find_config(context: ClaimContext, name: str) -> ModelConfig | None:
    return next((m for m in context.models if m.name == name), None)
