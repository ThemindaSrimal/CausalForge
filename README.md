# CausalForge

CausalForge is an autonomous causal-discovery laboratory. Meta Muse Spark chooses interventions through real API tool calls, an authoritative procedural simulator returns observations, and a local Bayesian model learns the hidden causal rule. A native WebGL interface visualizes the exact simulation state.

## What the demo proves

- Muse Spark makes scientific decisions through `run_experiment` tool calls.
- Experiments execute against hidden rules generated from arbitrary seeds.
- The local predictive model trains only on resulting observations.
- The browser renders backend-owned simulation state, not scripted animation.
- Hidden truth remains unavailable until **Reveal rule** is selected.
- Every live decision, observation, learner update, token count, and estimated cost is recorded.

## Quick start

Requirements:

- Python 3.10 or newer
- A browser with WebGL support
- A Meta Model API key for live mode

```bash
git clone <repository-url>
cd CausalForge
```

Create the local configuration:

```powershell
# Windows PowerShell
Copy-Item .env.example .env
```

```bash
# macOS or Linux
cp .env.example .env
```

Put your key in `.env`:

```dotenv
MODEL_API_KEY=your-meta-model-api-key
```

No external Python packages are required. This conventional command is still safe to run:

```bash
python -m pip install -r requirements.txt
```

Preflight and start the application:

```bash
python -m causalforge.demo
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000), enter any integer seed, select **Reset world**, and then select **Start agent**.

If port 8000 is occupied:

```bash
python -m causalforge.demo --port 8001
```

## Offline replay

The repository includes a clearly labeled replay exported from a successful real Muse Spark session. Start the app without an API key and select **Load recorded replay**. The header changes from **LIVE** to **RECORDED REPLAY**; replay mode never claims to be a live API run.

## Tests

```bash
python -m unittest discover -s tests -v
```

The paid integration test runs only when `MODEL_API_KEY` is configured. Without a key it is explicitly skipped; no fake API result is substituted.

## Evaluation

The locked evaluation ran 10 real Muse-directed experiments on each of 10 unseen procedural worlds:

- 10/10 runs completed
- 10/10 identified the exact active causal factors
- Mean coefficient RMSE: 0.019477
- Total evaluation cost: $0.71795890

See [EVALUATION_REPORT.md](EVALUATION_REPORT.md) for per-seed results. To repeat the paid evaluation:

```bash
python -m causalforge.evaluation
```

## Architecture

The repository is deliberately a modular monolith:

- `causalforge/` — simulator, learner, Muse adapter, orchestration, API server, evaluation, and replay
- `web/` — dependency-free native WebGL interface
- `tests/` — deterministic component, resilience, replay, and real API integration tests
- `replays/` — recorded real-session fallback evidence

See [ARCHITECTURE.md](ARCHITECTURE.md) for the system diagram and trust boundaries.

## Cost controls

Meta usage is estimated from provider-reported tokens. Each run has a configurable cap, while `runs/meta-spend.json` enforces a persistent project-level stop at $120—leaving a $30 margin below the $150 project ceiling.

## License

MIT — see [LICENSE](LICENSE).
