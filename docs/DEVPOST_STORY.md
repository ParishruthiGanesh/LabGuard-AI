## Inspiration

Two failure modes cost research time, and they are usually treated as separate problems. A run dies at 3am and nobody notices until morning. And — quieter, and far more expensive — a run succeeds perfectly while the conclusion drawn from it is wrong: one lucky seed, a baseline trained for a third as long, a checkpoint chosen on the test split, accuracy quoted on an 8%-positive benchmark.

They are the same problem seen from two sides. An experiment that ran correctly can still support the wrong conclusion, and an experiment whose health nobody watched cannot support any conclusion at all. LabGuard treats them as one workflow.

## What it does

You give LabGuard a research claim, your experiment configuration, your existing results, a compute budget and an autonomy policy. It then:

1. breaks the claim into measurable subclaims;
2. reads your configuration and finds the scientific loopholes in it — seed sensitivity, class imbalance, unfair baselines, cherry-picked checkpoints, misleading metrics, leakage, threshold artefacts;
3. plans the cheapest experiments that could actually settle them, and asks for approval when one is expensive;
4. queues the approved work asynchronously through Pub/Sub and runs it on a separate worker;
5. watches every run live for overfitting, divergence, stalls and corrupted checkpoints, repairing what its autonomy policy allows;
6. stops and escalates when a repair loop stops making progress;
7. audits the evidence, decides whether another test is worth it, and recurses;
8. issues a verdict with a reliability score where every number shows its arithmetic.

## The demo that matters

The bundled scenario is a claim worth rejecting: *"Model B performs better than Model A."* It is reported on one seed, with Model B trained 90 epochs against Model A's 25, and both checkpoints selected on the test split.

LabGuard's corrected comparison — five seeds, equal budget, validation-selected checkpoints — finds:

| Metric | Mean difference over 5 seeds | 95% interval | Seeds B wins |
| --- | ---: | --- | ---: |
| Accuracy | +0.0058 | [−0.0142, +0.0258] | 3 of 5 |
| Macro F1 | −0.0491 | [−0.0849, −0.0133] | **0 of 5** |
| Balanced accuracy | −0.0704 | [−0.0903, −0.0505] | **0 of 5** |

The class-wise breakdown says why: Model B's gain is confined to the majority class, while minority-class recall falls from 0.45 to 0.34. For a violence detector, that is the opposite of an improvement.

Along the way it detects overfitting in the reported run, recovers a submitted variant that genuinely diverges to NaN by scaling the learning rate inside declared bounds, and stops a corrupted-checkpoint job after three identical failures instead of retrying forever.

**Verdict: not sufficiently supported.** Overall claim confidence 29, with all seven dimensions showing the checks that produced them.

None of this is scripted. The imbalance, the divergence, the overfitting and the metric reversal are real properties of the data generator and the training loop, and the test suite asserts they are.

The same engine runs on your own claim. A well-controlled ablation we submitted through the form comes back **fragile at confidence 72**; a fair but noisy comparison comes back **not sufficiently supported at 50**. The score tracks how sound the setup actually is.

## How we built it

* **Gemini 3.5 Flash** for claim analysis, skeptical review, experiment planning, evidence interpretation and the verdict narrative.
* **Google Agent Development Kit** — each of the seven agents is an ADK `LlmAgent` with a strict `output_schema`, run through an ADK `Runner`.
* **Cloud Run** — three services: dashboard, API, and a private experiment worker that scales to zero.
* **Pub/Sub** — approved jobs are published, not executed inline, and push-delivered to the worker.
* **Firestore** — the shared state all seven agents read and write. They never talk to each other directly.
* **Cloud Storage** for reports and recorded curves, **Cloud Logging** for structured events, **Next.js 15**, **TypeScript**, **Tailwind CSS**, **FastAPI**, **NumPy**, **scikit-learn**.

## What makes it more than a chatbot

**The agents are given no tools.** Gemini reaches the system only through ADK agents with a strict output schema, and every response is a proposal that gets validated:

* It may *name* an action, only from a typed registry of 14, and only from the candidate set the planner already checked against the budget.
* Parameters are validated with pydantic and bound-checked — twice, once at planning and again in the worker before execution.
* It cannot produce a number. Every metric, confidence interval, per-class figure, health detection and reliability score is computed in Python.
* If its proposed verdict status disagrees with the measured one, the measurement wins and the disagreement is written to the ledger.

There is no path from model output to a shell, an `eval`, or a file write.

## Challenges we ran into

**Making the failures real.** The first version trained cross-entropy models and simply never produced a NaN — bounded gradients don't diverge. Rather than fake it, we added a squared-error-on-logit objective, which provably diverges above a critical learning rate. The NaN in the demo is genuine numerical divergence.

**Not lying in the UI.** An early build logged "applying approved recovery" for findings detected after a run had already finished — nothing had been applied. Another accused researchers of test-set checkpoint selection even when they had selected on validation, which dragged sound claims down to "not sufficiently supported". Both were found by running claims other than the bundled one, and both are fixed.

**Approving a repair that never happened.** Under safe-repair autonomy, a bounded learning-rate change needs a human decision. The job was held for approval — but the repair was never staged, so approving it re-ran the identical failing configuration and the loop detector blocked it. The researcher had approved a fix that did not exist. Repairs are now computed and staged before they are proposed, so the approval screen names the concrete change.

**Deploying.** The Cloud Run deployment surfaced four bugs that could only appear in a built image: a reserved Cloud Build substitution, two missing IAM grants, and a `next.config.ts` that made Next try to install TypeScript on every container start until the health check killed it. The last one is the reason the dashboard crash-looped for an hour.

## What we learned

The hard part of an autonomous agent is not making it act. It is deciding what it must never be allowed to decide. Every design choice here came down to drawing that line: the model proposes, and validated code disposes.

## What's next

* Adapters for real training frameworks, so LabGuard can watch PyTorch and JAX runs rather than its own loop.
* More loophole detectors: distribution shift over time, annotator agreement, multiple-comparison correction across reported ablations.
* Team mode — a shared reliability history across a lab's claims.
