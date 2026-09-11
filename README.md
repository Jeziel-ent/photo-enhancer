# Adinn 4K Image Enhancer

**Desktop software**, not a website. The user picks one or more images on
their own machine, a local AI pipeline enhances them on the local GPU, and
the result is saved back to disk — a single file for one input, or a ZIP
for multiple inputs. No server, no account, no external AI API.

**Core principle:** the selected photograph is the source of truth.
Enhancement only restores, denoises, deblurs, and upscales — it must never
invent, replace, remove, or hallucinate content (text, logos, faces, people,
vehicles, buildings, roads, signs, objects, geometry, surroundings).

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full layer
breakdown and the desktop-shell recommendation (pywebview).

## Layout

```
backend/            Application/API layer + desktop shell (see backend/README.md)
├── server.py       local HTTP API — POST /api/jobs, GET status/result, GET /health
├── jobs.py         single-worker processing queue (one GPU, one job at a time)
├── engine_adapter.py  the only module that imports image_enhancer/src/enhance.py
├── output_manager.py  single-file save / ZIP packaging
├── shell.py        pywebview desktop shell (proof of concept)
└── tests/, tests_integration/
frontend/           Desktop UI — React + Vite + Tailwind, rendered inside the desktop shell's window (placeholder; upload/queue/preview UI not yet built)
image_enhancer/     Image Enhancement Engine + local AI models + R&D
├── src/            enhance.py (production entry point) + experiment folders (f2/f3/g1-g9/pro/restore/swinir/f3_gate_exp)
├── tests/          regression scripts for the production pipeline
└── models/         RealESRGAN/SwinIR/Restormer weights (gitignored) — local, no external AI API
docs/               architecture and engine-audit notes
```

## Status

This is a standalone desktop image-enhancement product. Implemented so far:
the Image Enhancement Engine (`image_enhancer/`, unchanged production
pipeline plus its R&D history), the Application/API layer (`backend/`:
job queue, engine adapter, output manager, multipart parsing — see
`backend/README.md`), and a pywebview desktop-shell proof of concept
(`backend/shell.py`). Not yet built: the actual upload/queue/preview UI in
`frontend/` (currently a placeholder page) and packaging/distribution. See
`docs/ARCHITECTURE.md` for the full layer breakdown and shell rationale,
and `docs/ENGINE_AUDIT.md` for the engine's tuning history.
