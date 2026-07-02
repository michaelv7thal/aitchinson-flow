Review verdict

  I reviewed the uncommitted diff (train_for_sflm_bench.py, eqm.py, eqm_ae.py, fm_clr.py) plus the five new drive_*.sh orchestration scripts, with 7 finder angles
  and direct source verification. Nothing here corrupts the currently-running pipeline mid-flight, but there are 10 real issues — the most important being that
  the new batch-chunking is not behavior-preserving (despite the comment claiming exactness) and that the new DirichletFM_ep30_d30k alias silently joins the 
  default training queue.

  [
    {
      "file": "src/aitchinson_flow/models/eqm.py",
      "line": 425,
      "summary": "The chunking comment claims 'chunking is exact (the only batch-coupled term is the max-‖grad‖ early-stop, negligible)' — but with the default
  return_best=True the NAG loop selects ONE best iterate for the whole batch by batch-MEAN grad norm (eqm.py:491,501), and the g_min early stop is batch-max
  (eqm.py:494); chunking changes both statistics' population from B=256 to 16. Noise is also now drawn per-chunk, changing seeded RNG consumption order.",
      "failure_scenario": "Re-running the same seeded eval command on the same EqM-family checkpoint produces different samples and metrics than every
  pre-chunking run — the regenerated eval_all.json files in this working tree are silently not comparable to the committed numbers, so a metrics diff conflates
  the code change with a model change (phantom regression or masked real one)."
    },
    {
      "file": "src/aitchinson_flow/models/fm_clr.py",
      "line": 181,
      "summary": "FMonCLR.sample always integrates t from 0→1 regardless of x_init (no t_start), and the rewrite removed the c(γ) decay that previously made the
  field weak near data — so recovery_check.py's x_init-based protocol now feeds a nearly-clean perturbed input that the full-magnitude constant field fully
  transports as if it were t=0 noise.",
      "failure_scenario": "After drive_post.sh's --force retrain, FMonCLR's Δ@α craters for a reason unrelated to model quality (its sampler overwrites the
  conditioning input), while arms with α-matched start points keep theirs — the cross-arm recovery table becomes apples-to-oranges with no flag."
    },
    {
      "file": "scripts/train_for_sflm_bench.py",
      "line": 202,
      "summary": "DirichletFM_ep30_d30k is added to ARM_TO_MODEL, and ARMS = list(ARM_TO_MODEL) (line 212), so the alias joins the DEFAULT queue; its 30ep×30k
  recipe exists only in drive_dfm_extended.sh's CLI flags — nothing in the code enforces what the directory name claims.",
      "failure_scenario": "The documented full-queue invocation (no --only) trains a byte-identical duplicate of DirichletFM for hours of MIG time and writes
  runs/.../DirichletFM_ep30_d30k/ at whatever --epochs the caller passed — e.g. a 50-epoch run cited as the 30ep×30k extended-budget result."
    },
    {
      "file": "scripts/train_for_sflm_bench.py",
      "line": 359,
      "summary": "_model_cfg (which for EqMAE torch.load's the AE checkpoint and raises on a missing file) is called outside the per-stage try/except ladder (try
  starts ~line 370), and main() calls _train_arm bare — so a missing/corrupt AE ckpt crashes the entire multi-arm queue with no FAILED.json, violating the
  mark-failed-and-continue contract.",
      "failure_scenario": "drive_train.sh Stage A VAE fails → Stage B2 dies with FileNotFoundError before the ladder try; no FAILED.json is written (the census
  prints MISSING with no failure marker), and any multi-arm queue containing EqMAE never trains the arms after it."
    },
    {
      "file": "drive_train.sh",
      "line": 18,
      "summary": "`echo \"### [$(ts)] STAGE A exit=$?\"` always prints exit=0: the $(ts) command substitution runs first and clobbers $? before it expands
  (verified: `false; echo \"[$(ts)] exit=$?\"` prints 0). Same pattern at drive_train.sh:25,32; drive_post.sh:32,37,44,49; drive_dfm_extended.sh:21,31,40.",
      "failure_scenario": "A training stage crashes with exit 1 → the autonomous-run log says 'exit=0' for every stage; post-mortem of a failed pipeline run has
  no record of which stage failed (only drive_eval.sh's `|| echo FAILED` pattern reports real codes). Fix: capture rc=$? on its own line before the echo."
    },
    {
      "file": "scripts/eval_bpc.py",
      "line": 160,
      "summary": "eval_bpc.py's FMonCLR branch still hardcodes the OLD implied-x1 convention (pred_x1 = x_γ − gradient_lambda·v with γ=U^gamma_power, lines
  96–120) while fm_clr.py's rewrite changed the inverse to pred_x1 = x_t − (1−t)·v under uniform t — the eval-side mirror was never updated.",
      "failure_scenario": "Running eval_bpc.py on a new-code FMonCLR checkpoint applies lam=1.0 where the correct coefficient (1−t)≈0 for most of the γ-mass,
  over-subtracting the full-magnitude velocity — the reported surrogate BPC is garbage with no error or version guard."
    },
    {
      "file": "scripts/train_for_sflm_bench.py",
      "line": 266,
      "summary": "The rewritten FMonCLR cfg comment says 'no aux CE — see models/fm_clr.py', but fm_clr.py trains WITH the aux CE by default (lambda_ce > 0; its
  own new docstring calls the CE 'the one EqM-family term deliberately kept') — two comments written by this same diff directly contradict each other about what
  the arm trains.",
      "failure_scenario": "The bench script is the natural entry point; a reader reports FMonCLR as the CE-free textbook-Lipman control in the writeup,
  misattributing the token-anchor's effect — or 'fixes' the mismatch by zeroing lambda_ce, silently changing the already-trained arm's recipe."
    },
    {
      "file": "src/aitchinson_flow/models/eqm.py",
      "line": 426,
      "summary": "`getattr(self.cfg.eqm, \"sample_batch_chunk\", 16) or 16` is copy-pasted at 5 sites but the field does not exist on the EqM config dataclass — a
  phantom knob whose `or 16` also forecloses ever disabling chunking; chunk=16 therefore applies unconditionally (16× sequential sampler launches even at L=40
  where B=256 fits trivially). Meanwhile the sibling EqMLatent (in ARM_TO_MODEL and bench MODELS, same second-order grad path) got NO chunking at all, and SFLMEBM
  solved the same OOM with a different mechanism (chunk kwarg, L-adaptive default, empty_cache).",
      "failure_scenario": "Setting sample_batch_chunk via dataclasses.replace raises TypeError (unknown field); the L=40 leg of the autonomous chain silently runs
  ~16× more launch-bound NAG invocations across 3 EqM-family arms × 5 recovery alphas; an L256 EqMLatent eval still reproduces the exact NVML assert this diff
  was written to fix."
    },
    {
      "file": "drive_post.sh",
      "line": 21,
      "summary": "The three-stage chain (drive_post → drive_backfill → drive_dfm_extended) advances by grep-polling marker strings in log files the producers
  never write themselves — the filenames exist only in the launcher's redirect convention — with no timeout and no liveness check on the producer.",
      "failure_scenario": "A producer is OOM-killed / the box reboots / a relaunch uses a slightly different redirect name → the COMPLETE marker never appears and
  the downstream waiter sleep-loops forever, idling the MIG slice for days with no error. (The markers did land this run — the chain is past backfill — so this
  is a relaunch/reuse hazard, not a live failure.)"
    },
    {
      "file": "drive_dfm_extended.sh",
      "line": 33,
      "summary": "The extended arm's identity is patched post-hoc: only eval_all.json gets model_name relabeled (errors swallowed by 2>/dev/null); recovery.json
  and the checkpoint cfg still say DirichletFM; the arm is absent from drive_eval.sh/drive_backfill.sh's hardcoded ARMS lists; and backfill hardcodes --n 256
  --steps 200 while drive_eval.sh takes env-overridable N/STEPS.",
      "failure_scenario": "Any re-run of eval_all.py on this arm (manual backfill, post-fix re-eval) regenerates the JSON as model_name='DirichletFM', and the 
  aggregate table conflates or shadows the matched-budget row; a tier evaluated with non-default N then backfilled at n=256 mixes MC sample sizes across arms with
  no marker. A --label flag on eval_all.py (it already separates model_name from model_class internally) removes the whole hack."
    }
  ]

      "failure_scenario": "Any re-run of eval_all.py on this arm (manual backfill, post-fix re-eval) regenerates the JSON as model_name='DirichletFM', and the
  aggregate table conflates or shadows the matched-budget row; a tier evaluated with non-default N then backfilled at n=256 mixes MC sample sizes across arms with
  no marker. A --label flag on eval_all.py (it already separates model_name from model_class internally) removes the whole hack."
    }
  ]

  Notes

  - Most actionable now: findings 1–2 affect the interpretation of results being generated by the running pipeline (the modified eval_all.json files already
  reflect the chunked sampler), and finding 3 should be fixed before anyone runs the bench script without --only — a one-line fix is keeping the alias out of ARMS
  (or replacing it with a --run-suffix CLI flag; _train_arm already accepts out_dir).
  - Verified clean: the chunked recursive calls forward every parameter of their own signatures; eval scripts rebuild config from the checkpoint payload, so
  FMonCLR's source_sigma=1.0 survives into eval and drive_post.sh's retrain-then-eval plan is sound; the new EqMAE copy-all-valid-fields load fix is correct and
  fixes a genuine VAE-as-AE mis-load.
  - Honorable mention (didn't make the cap): source_sigma=1.0 lives only in the bench script's _model_cfg — an FMonCLR run launched via main.py or a sweep trains
  the constant-target loss on the EqM-tuned σ=0.1 near-point-mass source, the exact pathology the change exists to avoid. Consider moving it into the model or a
  config default.

  I made no changes to the working tree — review only, per the running session.