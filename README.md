# ASAS-SN Light Curve Viewer + Fitter

A local web app for viewing and fitting ASAS-SN photometry data with interactive Plotly charts.

## Prerequisites

You need **conda** installed (any of Miniconda, Anaconda, or Miniforge). If you don't have it yet, get [Miniforge](https://github.com/conda-forge/miniforge#install) — the installer below will find it automatically.

## Installation

```bash
git clone https://github.com/dhvanildesai/ass-viewer-fitter.git
cd ass-viewer-fitter
./install.sh
```

`install.sh` creates an isolated conda environment (`asassn-viewer`, Python 3.13) and installs the required packages (Flask, numpy, pandas, scipy, astropy) into it via pip. It also generates `run_ass.sh`, a launch script that activates that environment for you.

> `run_ass.sh` is generated fresh by `install.sh` on your machine (it embeds local conda paths), so it's gitignored rather than committed — always run `install.sh` first on a new machine, don't copy `run_ass.sh` from elsewhere.

## Running the App

```bash
./run_ass.sh
```

This opens the app in your browser at `http://127.0.0.1:5050`.

Optional flags (pass them straight through, e.g. `./run_ass.sh --port 8080`):
- `--port 8080` — use a different port
- `--no-browser` — don't auto-open a browser tab

## Basic Usage

### Load light curves
- **Upload LCs** — drag and drop (or select) one or more `.dat` ASAS-SN light curve files. The band (g or V) is detected from the filename (`..._ASASSN_g.dat` / `..._ASASSN_V.dat`), and multiple files for the same object are merged automatically.
- **Master CSV** — load a catalog CSV of objects at once. Point it at the directory containing your `.dat` files (looked up as `[ID]_ASASSN_g.dat` / `[ID]_ASASSN_V.dat`), and optionally an output CSV path and a template directory.

### Fit a light curve
1. Set the **Template Directory** (right panel, top) and click **Load Templates**.
2. Pick a **Subtype** and the **Band(s)** to fit.
3. Set the peak flux/time guess — either type values directly, or click **Pick Peak from Plot** and click the peak on the chart.
4. Click **Run Fit**. Right-click a point on the chart to remove an outlier and refit; shift-drag to remove a whole region.
5. Set an **Output CSV** path and click **Save This Object** to save your result. Reloading the same output CSV later restores your saved progress.

### Navigate objects
Use the `‹`/`›` buttons, or type a name in "Jump to name…" and press Enter.

That covers the basics — the right-hand panel has more controls (Monte Carlo error estimation, flags, redshift/extinction, baseline correction, etc.) worth exploring later on.
