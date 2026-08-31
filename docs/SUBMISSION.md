# Devpost submission — copy/paste

## Project name
```
LabGuard AI
```

## Elevator pitch (200 char max)
```
An autonomous research agent that challenges a scientific claim, runs the experiments that could disprove it, repairs the runs that break, and returns an evidence-backed verdict with every check shown.
```

## Built with (tags)
```
gemini, google-adk, vertex-ai, cloud-run, firestore, pub-sub, cloud-storage, cloud-logging,
python, fastapi, numpy, scikit-learn, nextjs, react, typescript, tailwindcss, docker, cloud-build
```

## "Try it out" links
```
https://labguard-dashboard-nlpoi32vfq-uc.a.run.app
https://github.com/ParishruthiGanesh/LabGuard-AI/tree/claude/labguard-ai-platform-4vgw8p
```

## Category
**Taskmaster**

## Which Google SDK did you use?
**Agent Development Kit (ADK)** and **Google GenAI SDK (google-genai)**

## Which Google Cloud Service(s) did you use?
**Cloud Run · Firestore · Pub/Sub · Cloud Storage · Vertex AI · Cloud Logging · Artifact Registry · Cloud Build**

## Did you add Reproducible Testing instructions to your README?
**Yes**

## Hosted project URL
```
https://labguard-dashboard-nlpoi32vfq-uc.a.run.app
```

## Testing instructions (for judges, not public)
```
No setup or credentials needed — the hosted dashboard is public.

1. Open the dashboard. Leave "Managed autonomy" selected. Click "Start verification".
2. Open "Claim map": eight measurable subclaims and eleven loopholes, all derived from the
   submitted configuration.
3. Open "Experiment plan". The five-seed retrain costs 8 units against a 6-unit threshold, so it
   stops for approval. Nothing has run — every job is held at awaiting_approval. Click Approve.
4. Open "Live run health" and watch the training curves stream. Three incidents appear:
   overfitting in the reported run, a NaN divergence that is repaired by a bounded learning-rate
   change, and a corrupted checkpoint that is stopped after three identical failures.
5. Open "Final report": verdict "not sufficiently supported", with every reliability check listed
   with its weight, outcome and the number behind it.

To run it locally instead (no cloud project, no API key):
    make setup && make api      # terminal 1
    make dashboard              # terminal 2, then http://localhost:3000
    make test                   # 74 tests, ~35s

To verify the synthetic benchmark's weaknesses are real rather than scripted:
    cd backend && .venv/bin/python -m pytest tests/test_experiments.py -v
```

## Architecture diagram
Upload `docs/assets/architecture.png`.
