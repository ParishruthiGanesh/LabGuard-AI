# Four-minute demo script

Total: 4:00. Times are cumulative. Run `python scripts/seed_demo.py --no-approve`
beforehand if you want a pre-warmed claim to fall back on.

**Setup:** backend on `:8080`, dashboard on `:3000`, browser at
`http://localhost:3000`, one tab, zoom 110%.

Start the backend with `SIMULATED_EPOCH_DELAY=0.03` for this walkthrough.
Training is fast enough that a run finishes before the dashboard polls twice;
the pacing lets the audience watch the curves build. It adds roughly a minute
to the whole run and changes no result.

---

## 0:00 — 0:30 · The problem

> "Researchers lose weeks to failed experiments. Worse, they sometimes publish
> conclusions from experiments that succeeded technically and failed
> scientifically. LabGuard protects both sides: it watches every run and
> repairs what it can, while continuously trying to falsify the claim behind
> it. It doesn't stop when the code runs. It stops when the evidence can be
> trusted."

Point at the launcher. The claim is on screen:

> *"Model B performs better than Model A on the violence-detection benchmark."*

> "Reported on one seed. Model B trained 90 epochs, Model A 25. Both
> checkpoints picked on the test split. Every one of those is a real property
> of the configuration — nothing here is scripted."

Leave **Managed autonomy** selected. Click **Start verification**.

## 0:30 — 1:10 · Decompose and challenge

Land on **Overview**, then open **Claim map**.

> "The Claim Analyst turned one sentence into eight measurable subclaims. The
> Scientific Skeptic found eleven loopholes by reading the submitted
> configuration — an 8% positive rate makes accuracy nearly uninformative, the
> arms had unequal training budgets, checkpoints were chosen on test, and it's
> a single seed."

Scroll to alternative explanations.

> "It also wrote down the rival explanations it has to rule out."

## 1:10 — 1:45 · Plan and approve

Open **Experiment plan**.

> "The planner picked eight actions for 12.6 compute units — cheapest decisive
> checks first. Every action comes from a typed registry with validated
> parameters. The model can name an action; it can never issue a command."

Point at the amber banner.

> "The five-seed retrain costs 8 units against a 6-unit threshold, so it stops
> for me. Nothing has run — every job is holding at `awaiting_approval`."

Click **Approve 12.60 units**.

## 1:45 — 2:40 · Watch the runs

Open **Live run health** immediately.

> "These jobs went out over Pub/Sub. The worker is a separate Cloud Run
> service — the curves you're seeing stream from real training runs, epoch by
> epoch."

Click through the run selector and narrate the three incidents:

1. **`inspect_training_curve` · Model B** — validation loss climbing away from
   its best while training loss keeps falling.
   > "Overfitting, detected live. This one is a diagnostic replay of the
   > reported run, so RunMedic records it rather than interrupting the
   > measurement — and applies early stopping to the verification runs instead."

2. **`inspect_training_curve` · fast-LR variant** — NaN, then recovered.
   > "This variant genuinely diverges — squared error on the logit at learning
   > rate 2.4. RunMedic caught the NaN, scaled the rate down inside the bounds
   > declared on the action, re-queued it, and it completed. Two attempts,
   > fully audited."

3. **`resume_from_checkpoint`** — blocked.
   > "The reported checkpoint fails its integrity check. LabGuard repaired and
   > retried, got the identical failure signature, and stopped at three
   > attempts rather than burning budget on a failure that wasn't changing."

## 2:40 — 3:10 · Recursion

Open **Queue**, then **Overview**.

> "The Evidence Auditor found the threshold subclaim still untested, so it
> planned a second round on its own and ran it. That's the recursion: it keeps
> going until another experiment couldn't change a conclusion, the budget runs
> out, or it needs me."

## 3:10 — 3:50 · The verdict

Open **Final report**.

> "**Not sufficiently supported.** Across five seeds under an equalised budget,
> the accuracy difference averages +0.006 with a 95% interval that contains
> zero. Macro F1 is *negative* — Model B loses on all five seeds. So does
> balanced accuracy. The class-wise breakdown says why: the gain is entirely in
> the majority class, and minority recall drops from 0.45 to 0.34. For a
> violence detector, that's the opposite of an improvement."

Scroll through the reliability checks.

> "Every score is a weighted pass rate over named checks, each showing its
> arithmetic. Data integrity is 100 — there's genuinely no leakage, and the
> detector was proven on a positive control. Statistical stability is 0, and
> you can see exactly which three checks failed. Gemini writes the narrative;
> it is never allowed to produce a number, and if its proposed status disagrees
> with the measured one, the measurement wins and the disagreement is logged."

Click **Download the full report**.

## 3:50 — 4:00 · Close

> "Two loops, one shared state: challenge the claim, protect the run, trust the
> result. Gemini 3.5 Flash through Google ADK, Firestore, Pub/Sub and three
> Cloud Run services — and the whole thing runs offline on a deterministic rule
> engine for judging, with the identical state machine."

---

## If something goes wrong

| Symptom | Do this |
| --- | --- |
| Dashboard shows a connection error | The backend restarted. State is in memory in demo mode; start a fresh claim. |
| A run seems stuck | Check **Queue**. `blocked_loop` on `resume_from_checkpoint` is the intended demo behaviour, not a hang. |
| Verdict is slow | The whole workflow is ~20s of real training. Fill with the Claim map. |
| No network | Demo mode needs none. Everything above works offline. |

## Talking points if asked

* **"Is the model just making this up?"** — Open the Evidence ledger. Every row
  names the agent, the reason, the input, the result and the decision. Metrics,
  intervals, per-class figures and health detections are all computed in
  Python. The model's proposed verdict status is overruled by the measured one.
* **"What if Gemini is down?"** — Each agent falls back to the rule engine for
  that step. The workflow completes; only the prose changes.
* **"Could it run something dangerous?"** — There is no path from model output
  to a shell or an `eval`. Actions are looked up by name in a fixed registry
  and their parameters are bound-checked before execution.
* **"Is the demo data rigged?"** — The generator and training loop are in
  `backend/labguard/experiments/`. `pytest tests/test_experiments.py` asserts
  the weaknesses are real properties of the maths, not scripted outputs.
