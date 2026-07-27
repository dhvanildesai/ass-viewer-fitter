"""
ASAS-SN Light Curve Viewer + Fitter v2
Flask-based local web app with interactive Plotly charts.
"""

import os, io, json, re, copy
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file, abort
import webbrowser, threading, time

from scipy.interpolate import RectBivariateSpline, interp1d
from scipy.optimize import minimize

try:
    from astropy.time import Time as AstropyTime
    from astropy.cosmology import FlatLambdaCDM
    HAS_ASTROPY = True
    _COSMO = FlatLambdaCDM(H0=70, Om0=0.3)
except ImportError:
    HAS_ASTROPY = False
    _COSMO = None

app = Flask(__name__, template_folder=".")
app.secret_key = "asassn_viewer_secret_2024"
SCRIPT_DIR = Path(__file__).parent.resolve()
FLUX_TO_MAG_CONST = 2.5 / np.log(10)

DEFAULTS = {
    "template_dir": "/Users/dhvanildesai/Desktop/UH_IfA/Thesis/gband_rates/templates",
    "output_csv":   "/Users/dhvanildesai/Desktop/UH_IfA/Thesis/gband_rates/sn_data/TEST_fit_results.csv",
    "lc_dir":       "/Users/dhvanildesai/Desktop/UH_IfA/Thesis/gband_rates/sn_data/ASASSN_SN_lightcurves/all_lcs/lcs",
}

# ── Template catalog constants ────────────────────────────────────────────────
TEMPLATE_SUBDIR_MAP = {
    'Ia_norm':                'norm_flux_templates_for_Ia_norm',
    'Ia_02es':                'Ia_02es_templates',
    'Ia_91T':                 'Ia_91T_templates',
    'Iax':                    'Iax_templates',
    'Ia_03fg_2012dn':         'Ia_03fg_templates',
    'Ia_03fg_2021zny':        'Ia_03fg_templates',
    'Ia_CSM_2018evt_2022erq': 'Ia_CSM_templates',
    'Ia_CSM_2022esa':         'Ia_CSM_templates',
    'II_ASASSN14jb':          'CCSNe_templates_Vband_for_testing',
    'II_SN1987A':             'CCSNe_templates_Vband_for_testing',
    'II_SN2004et':            'CCSNe_templates_Vband_for_testing',
    'II_SN2013ej':            'CCSNe_templates_Vband_for_testing',
    'II_SN2014G':             'CCSNe_templates_Vband_for_testing',
    'IIb_SN2008ax':           'CCSNe_templates_Vband_for_testing',
    'Ib_Ibn_iPTF13bvn':       'CCSNe_templates_Vband_for_testing',
    'Ic_SN2004gt':            'CCSNe_templates_Vband_for_testing',
    'Ic_CSM_2022esa':         'CCSNe_templates_Vband_for_testing',
    'IcBL_SN2009bb':          'CCSNe_templates_Vband_for_testing',
    'IIn_SN2006aa':           'CCSNe_templates_Vband_for_testing',
    'IIn_SN2007pk':           'CCSNe_templates_Vband_for_testing',
    'IIn_SN2009ip':           'CCSNe_templates_Vband_for_testing',
    'IIn_SN2010al':           'CCSNe_templates_Vband_for_testing',
    'IIn_SN2011ht':           'CCSNe_templates_Vband_for_testing',
    'SLSN_ASASSN15lh':        'CCSNe_templates_Vband_for_testing',
    'SLSN_Gaia17biu':         'CCSNe_templates_Vband_for_testing',
    'SLSN_SN2018bgv':         'CCSNe_templates_Vband_for_testing',
    'CV':                     'CV_templates',
    # TDE templates — one subtype per object, subdir/file both named after the object
    'ASASSN-14ae':            'ASASSN-14ae_templates',
    'ASASSN-14li':            'ASASSN-14li_templates',
    'ASASSN-15oi':            'ASASSN-15oi_templates',
    'TDE2018dyb':             'TDE2018dyb_templates',
    'TDE2018fyk':             'TDE2018fyk_templates',
    'TDE2018hyz':             'TDE2018hyz_templates',
    'TDE2018zr':              'TDE2018zr_templates',
    'TDE2019ahk':             'TDE2019ahk_templates',
    'TDE2019azh':             'TDE2019azh_templates',
    'TDE2019dsg':             'TDE2019dsg_templates',
    'TDE2019qiz':             'TDE2019qiz_templates',
    'TDE2019vcb':             'TDE2019vcb_templates',
    'TDE2020afhd':            'TDE2020afhd_templates',
    'TDE2020neh':             'TDE2020neh_templates',
    'TDE2020nov':             'TDE2020nov_templates',
    'TDE2020vwl':             'TDE2020vwl_templates',
    'TDE2022bdw':             'TDE2022bdw_templates',
    'TDE2022dbl':             'TDE2022dbl_templates',
    'TDE2022dsb':             'TDE2022dsb_templates',
    'TDE2022hvp':             'TDE2022hvp_templates',
    'TDE2022lri':             'TDE2022lri_templates',
    'TDE2023clx':             'TDE2023clx_templates',
    'TDE2023mhs':             'TDE2023mhs_templates',
    'TDE2024as':              'TDE2024as_templates',
    'TDE2024pvu':             'TDE2024pvu_templates',
    'TDE2024tvd':             'TDE2024tvd_templates',
    'TDE2025aarm':            'TDE2025aarm_templates',
}

IA_NORM_S_VALUES = np.array([1.15, 0.90, 0.75, 0.60, 0.50, 0.40, 0.32, 0.25, 0.20, 0.15, 0.10])
IA_NORM_M_VALUES = np.array([-19.50,-19.25,-19.00,-18.62,-18.30,-17.92,-17.56,-17.25,-17.00,-16.74,-16.46])

# ── Template loading ──────────────────────────────────────────────────────────

def load_template_catalog(template_dir: str) -> dict:
    """Load all templates. Returns {subtype: info_dict}. interp functions NOT JSON-safe."""
    catalog = {}
    for subtype, subdir in TEMPLATE_SUBDIR_MAP.items():
        full_dir = os.path.join(template_dir, subdir)
        if not os.path.isdir(full_dir):
            continue

        if subtype == 'Ia_norm':
            common_t = np.linspace(-40, 130, 1000)
            flux_grid, s_valid, M_valid = [], [], []
            for s, M in zip(IA_NORM_S_VALUES, IA_NORM_M_VALUES):
                fpath = os.path.join(full_dir, f"Ia_norm_flux_template_s_{s:.2f}_M_{M:.2f}.txt")
                if not os.path.isfile(fpath):
                    continue
                try:
                    tdf = pd.read_csv(fpath, sep=r'\s+', comment='#')
                    col = 'flux_g' if 'flux_g' in tdf.columns else tdf.columns[1]
                    flux_grid.append(np.interp(common_t, tdf['time'].values, tdf[col].values, left=0, right=0))
                    s_valid.append(s); M_valid.append(M)
                except Exception:
                    continue
            if len(flux_grid) >= 2:
                s_arr = np.array(s_valid)
                si = np.argsort(s_arr)
                s_s = s_arr[si]
                g_s = np.array(flux_grid).T[:, si]
                try:
                    catalog['Ia_norm'] = {
                        'type': '2D',
                        'interp': RectBivariateSpline(common_t, s_s, g_s, kx=3, ky=3),
                        's_values': s_s,
                        'M_values': np.array(M_valid)[si],
                    }
                except Exception:
                    pass
        else:
            fpath = os.path.join(full_dir, f"{subtype}_flux_template.txt")
            if not os.path.isfile(fpath):
                continue
            try:
                tdf = pd.read_csv(fpath, sep=r'\s+', comment='#')
                t_vals = tdf['time'].values
                ttype = 'CV' if subtype == 'CV' else '1D'
                entry = {'type': ttype}
                for bcol in ('flux_g', 'flux_V'):
                    if bcol in tdf.columns:
                        bkey = bcol.split('_')[1]  # 'g' or 'V'
                        entry[f'interp_{bkey}'] = interp1d(
                            t_vals, tdf[bcol].values,
                            kind='linear', bounds_error=False, fill_value=0.0)
                if 'interp_g' not in entry and 'interp_V' not in entry:
                    col2 = tdf.columns[1]
                    fn = interp1d(t_vals, tdf[col2].values, kind='linear', bounds_error=False, fill_value=0.0)
                    entry['interp_g'] = entry['interp_V'] = fn
                if 'interp_g' not in entry: entry['interp_g'] = entry['interp_V']
                if 'interp_V' not in entry: entry['interp_V'] = entry['interp_g']
                catalog[subtype] = entry
            except Exception:
                continue
    return catalog


def _get_interp(tinfo: dict, band: str):
    if tinfo['type'] == '2D':
        return tinfo['interp']
    return tinfo.get(f'interp_{band}') or tinfo.get('interp_g')


# ── Model + chi-squared ───────────────────────────────────────────────────────

def model_fluxes(params, times, redshift, tinfo, band='g'):
    pf, pt = params[0], params[1]
    ttype = tinfo['type']
    z = max(float(redshift) if (redshift is not None and np.isfinite(float(redshift))) else 0.0, 0.0)

    if ttype == 'CV':
        stretch = max(params[2], 0.01) if len(params) > 2 else 1.0
        rel_t = (times - pt) / (stretch * (1 + z))
        return pf * _get_interp(tinfo, band)(rel_t)

    rel_t = (times - pt) / (1 + z)

    if ttype == '2D':
        sBV = params[2] if len(params) > 2 else 0.5
        return pf * tinfo['interp'](rel_t, sBV, grid=False)

    return pf * _get_interp(tinfo, band)(rel_t)


def chi_sq(params, times, fluxes, errors, redshift, tinfo, band):
    m = model_fluxes(params, times, redshift, tinfo, band)
    return float(np.sum(((fluxes - m) / errors) ** 2))


def n_params_for(tinfo):
    return 3 if tinfo['type'] in ('2D', 'CV') else 2


def _default_bounds(pf_guess, pt_guess, flux_delta, time_delta, tinfo, sBV_lo=0.10, sBV_hi=1.15):
    fl = max(pf_guess - flux_delta, 1e-9)
    fh = pf_guess + flux_delta
    tl = pt_guess - time_delta
    th = pt_guess + time_delta
    bounds = [(fl, fh), (tl, th)]
    ttype = tinfo['type']
    if ttype == '2D':
        s_values = tinfo.get('s_values', [0.10, 1.15])
        bounds.append((float(np.min(s_values)), float(np.max(s_values))))
    elif ttype == 'CV':
        bounds.append((0.5, 2.0))
    return bounds


def fit_with_clipping(times, fluxes, errors, redshift, tinfo, band,
                      p0, bounds, use_clipping=True):
    """Iterative sigma-clip then final fit. Returns (scipy_result, kept_mask)."""
    mask = np.ones(len(times), dtype=bool)
    np_fit = n_params_for(tinfo)

    if use_clipping:
        for sigma in [15.0, 5.0]:
            if mask.sum() <= np_fit:
                break
            r = minimize(chi_sq, p0,
                         args=(times[mask], fluxes[mask], errors[mask], redshift, tinfo, band),
                         method='L-BFGS-B', bounds=bounds)
            if not r.success:
                continue
            m_model = model_fluxes(r.x, times[mask], redshift, tinfo, band)
            res_sigma = np.abs(fluxes[mask] - m_model) / errors[mask]
            bad_local = res_sigma > sigma
            idx_all = np.where(mask)[0][bad_local]
            mask[idx_all] = False

    if mask.sum() <= np_fit:
        return None, mask

    result = minimize(chi_sq, p0,
                      args=(times[mask], fluxes[mask], errors[mask], redshift, tinfo, band),
                      method='L-BFGS-B', bounds=bounds)
    return result, mask


def run_monte_carlo(times, fluxes, errors, redshift, tinfo, band,
                    p_best, bounds, n_mc=250):
    """Returns array of shape (n_successful, n_params) or None."""
    samples = []
    for _ in range(n_mc):
        fake = np.random.normal(fluxes, errors)
        r = minimize(chi_sq, p_best,
                     args=(times, fake, errors, redshift, tinfo, band),
                     method='L-BFGS-B', bounds=bounds)
        if r.success:
            samples.append(r.x.tolist())
    return np.array(samples) if samples else None


# ── Cosmology ─────────────────────────────────────────────────────────────────

def compute_dist_mod(redshift):
    if not HAS_ASTROPY or redshift is None:
        return float('nan')
    try:
        z = float(redshift)
        if not np.isfinite(z) or z <= 0:
            return float('nan')
        return float(_COSMO.distmod(z=z).value)
    except Exception:
        return float('nan')


# ── Conversion helpers ────────────────────────────────────────────────────────

def mag_to_flux(mag, zp=3631.0):
    return zp * 10 ** (-mag / 2.5) * 1000.0

def flux_to_mag_scalar(flux_mJy, zp=3631.0):
    try:
        f = float(flux_mJy)
        if not np.isfinite(f) or f <= 0: return float('nan')
        return -2.5 * np.log10(f / 1000.0 / zp)
    except (TypeError, ValueError):
        return float('nan')

def ut_to_jd(ut_str):
    if not HAS_ASTROPY: return None
    s = re.sub(r'\.\d+$', '', str(ut_str).strip())
    try:
        return float(AstropyTime(s, format='iso', scale='utc').jd)
    except Exception:
        return None

def _safe(v):
    if v is None: return None
    try:
        f = float(v)
        return None if not np.isfinite(f) else round(f, 6)
    except (TypeError, ValueError):
        return None

def _safe4(v):
    s = _safe(v); return None if s is None else round(s, 4)

def _nan_or(v):
    """Return v if finite, else None."""
    if v is None: return None
    try:
        return None if not np.isfinite(float(v)) else v
    except (TypeError, ValueError):
        return None


# ── State ──────────────────────────────────────────────────────────────────────

# Single unified list shared by viewer and fitter
OBJECTS: list = []

STATE = {"current": 0}

FIT_STATE = {
    "current":          0,
    "results":          {},    # name -> serialisable result dict
    "completed":        set(),
    "template_dir":     "",
    "output_csv":       "",
    "template_catalog": None,  # loaded lazily; holds interp objects
    "n_mc":             250,
}


def _get_catalog():
    tc = FIT_STATE["template_catalog"]
    if tc is None and FIT_STATE["template_dir"]:
        tc = load_template_catalog(FIT_STATE["template_dir"])
        FIT_STATE["template_catalog"] = tc
    return tc


def _make_object(name, g_file=None, V_file=None, band=None, meta=None):
    b = band or ("V" if (V_file and not g_file) else "g")
    return {
        "name":            name,
        "g_file":          g_file,
        "V_file":          V_file,
        "band":            b,
        "active_bands":    "primary",   # "primary" or "both"
        "redshift":        None,
        "dist_mod":        None,
        "k_corr":          0.0,
        "gal_ext":         0.0,
        "disc_jd":         None,
        "type_to_use":     None,
        "prior_sub_type":  None,
        "prior_peak_time": None,
        "prior_app_mag":   None,
        "prior_sBV":       None,
        "meta":            meta or {},
        # modifications
        "removed_jds":     [],
        "v_scale":         1.0,
        "error_inflate":   0.0,
        "baseline_shifts": {},
        "flag":            None,
        # undo
        "undo_stack":      [],
    }


def _snap(obj):
    return {
        "removed_jds":     copy.copy(obj["removed_jds"]),
        "v_scale":         obj["v_scale"],
        "error_inflate":   obj["error_inflate"],
        "baseline_shifts": copy.copy(obj["baseline_shifts"]),
        "active_bands":    obj["active_bands"],
        "flag":            obj["flag"],
        "fit_result":      copy.deepcopy(FIT_STATE["results"].get(obj["name"])),
    }

def _push_undo(obj):
    obj["undo_stack"].append(_snap(obj))
    if len(obj["undo_stack"]) > 20:
        obj["undo_stack"].pop(0)

def _pop_undo(obj):
    if not obj["undo_stack"]: return False
    s = obj["undo_stack"].pop()
    obj["removed_jds"]     = s["removed_jds"]
    obj["v_scale"]         = s["v_scale"]
    obj["error_inflate"]   = s["error_inflate"]
    obj["baseline_shifts"] = s["baseline_shifts"]
    obj["active_bands"]    = s["active_bands"]
    obj["flag"]            = s["flag"]
    name = obj["name"]
    if s["fit_result"] is not None:
        FIT_STATE["results"][name] = s["fit_result"]
    else:
        FIT_STATE["results"].pop(name, None)
    return True

def _revert(obj):
    obj["removed_jds"]     = []
    obj["v_scale"]         = 1.0
    obj["error_inflate"]   = 0.0
    obj["baseline_shifts"] = {}
    obj["active_bands"]    = "primary"
    obj["flag"]            = None
    FIT_STATE["results"].pop(obj["name"], None)

def _find_obj(name):
    for o in OBJECTS:
        if o["name"] == name: return o
    return None


# ── LC parsing ────────────────────────────────────────────────────────────────

def parse_ut_date(ut_str: str) -> str:
    try:
        if "." in ut_str:
            dp, frac = ut_str.split('.', 1)
            ff = float("0." + frac)
            h = int(ff*24); m = int((ff*24-h)*60); s = int(((ff*24-h)*60-m)*60)
            return f"{dp} {h:02d}:{m:02d}:{s:02d}"
        return ut_str
    except Exception:
        return ut_str

def parse_lc_file(filepath: str, band_hint: str = None) -> Optional[pd.DataFrame]:
    rows = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue
                parts = line.split()
                if len(parts) < 8: continue
                try:
                    jd       = float(parts[0])
                    hjd      = float(parts[1])
                    ut       = parse_ut_date(parts[2])
                    img      = parts[3] if len(parts) > 3 else 'unknown'
                    m        = re.search(r'(b[a-zA-Z])', img)
                    cam      = m.group(1) if m else 'unk'
                    mag_raw  = parts[7]
                    is_upper = mag_raw.startswith('>')
                    mag_val  = float(mag_raw.lstrip('>'))
                    mag_err  = float(parts[8]) if len(parts) > 8 else 99.99
                    flux     = float(parts[11]) if len(parts) > 11 else float('nan')
                    flxerr   = float(parts[12]) if len(parts) > 12 else float('nan')
                    rows.append({
                        'JD': jd, 'HJD': hjd, 'UT_date': ut,
                        'IMAGE': img, 'camera': cam,
                        'mag': mag_val, 'mag_err': mag_err,
                        'upper_limit': is_upper or mag_err >= 99,
                        'flux': flux, 'flux_err': flxerr,
                        'flux_orig': flux, 'flux_err_orig': flxerr,
                    })
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None
    if not rows: return None
    df = pd.DataFrame(rows)
    if band_hint: df['band'] = band_hint
    return df


def load_combined_lc(obj) -> Optional[pd.DataFrame]:
    """Load raw LC data for an object (both bands if active_bands == 'both')."""
    band     = obj.get("band", "g")
    act      = obj.get("active_bands", "primary")
    dfs      = []

    pf = obj.get(f"{band}_file")
    if pf and os.path.isfile(pf):
        df = parse_lc_file(pf, band_hint=band)
        if df is not None and not df.empty: dfs.append(df)

    if act == "both":
        ob = "V" if band == "g" else "g"
        of = obj.get(f"{ob}_file")
        if of and os.path.isfile(of):
            df2 = parse_lc_file(of, band_hint=ob)
            if df2 is not None and not df2.empty: dfs.append(df2)

    if not dfs: return None
    return pd.concat(dfs, ignore_index=True).sort_values("JD").reset_index(drop=True)


def load_all_bands_lc(obj) -> Optional[pd.DataFrame]:
    """Load raw LC data for BOTH bands unconditionally, for display purposes —
    independent of which band(s) are 'active' for fitting (active_bands may
    restrict fitting to just one band while the other still has real data)."""
    dfs = []
    for b in ("g", "V"):
        fp = obj.get(f"{b}_file")
        if fp and os.path.isfile(fp):
            df = parse_lc_file(fp, band_hint=b)
            if df is not None and not df.empty: dfs.append(df)
    if not dfs: return None
    return pd.concat(dfs, ignore_index=True).sort_values("JD").reset_index(drop=True)


def apply_modifications(df: pd.DataFrame, obj: dict) -> pd.DataFrame:
    if df is None or df.empty: return df
    df = df.copy()
    v_scale = obj.get("v_scale", 1.0)
    if v_scale != 1.0 and "band" in df.columns:
        vm = df["band"] == "V"
        if vm.any():
            df.loc[vm, "flux"]     = df.loc[vm, "flux_orig"]     * v_scale
            df.loc[vm, "flux_err"] = df.loc[vm, "flux_err_orig"] * v_scale
    for cam, shift in obj.get("baseline_shifts", {}).items():
        cm = df["camera"] == cam
        if cm.any():
            df.loc[cm, "flux"] -= shift
    inflate = obj.get("error_inflate", 0.0)
    if inflate > 0:
        df["flux_err"] = np.sqrt(df["flux_err"]**2 + inflate**2)
    removed = obj.get("removed_jds", [])
    if removed:
        mask = np.ones(len(df), dtype=bool)
        for rjd in removed:
            mask &= ~np.isclose(df["JD"], float(rjd), rtol=0, atol=1e-5)
        df = df[mask]
    return df


def get_valid_fit_data(obj, jd_min=None, jd_max=None):
    """Returns (df_raw, df_modified, df_window) — df_window is ready for fitting."""
    df_raw = load_combined_lc(obj)
    if df_raw is None or df_raw.empty: return None, None, None
    df_mod = apply_modifications(df_raw, obj)
    ok = (
        df_mod["flux"].notna() &
        np.isfinite(df_mod["flux"].values) &
        (df_mod["flux"] != 99.990) &
        df_mod["flux_err"].notna() &
        np.isfinite(df_mod["flux_err"].values) &
        (df_mod["flux_err"] > 0)
    )
    valid = df_mod[ok].copy()
    if jd_min is not None: valid = valid[valid["JD"] >= jd_min]
    if jd_max is not None: valid = valid[valid["JD"] <= jd_max]
    return df_raw, df_mod, valid


def compute_abs_mag(app_mag, dist_mod, k_corr=0.0, gal_ext=0.0):
    try:
        a = float(app_mag); d = float(dist_mod)
        if not np.isfinite(a) or not np.isfinite(d): return float('nan')
        return a - d - float(k_corr or 0) - float(gal_ext or 0)
    except (TypeError, ValueError):
        return float('nan')


# ── Auto-guess ─────────────────────────────────────────────────────────────────

def auto_guesses(obj, jd_min=None, jd_max=None, catalog=None):
    _, _, valid = get_valid_fit_data(obj, jd_min, jd_max)
    pf_guess = float('nan'); pt_guess = float('nan')

    if valid is not None and not valid.empty:
        pf_guess = float(valid["flux"].max())
        pt_guess = float(valid.loc[valid["flux"].idxmax(), "JD"])

    disc_jd = obj.get("disc_jd")
    prior_pt = obj.get("prior_peak_time")
    if prior_pt is not None and np.isfinite(float(prior_pt)):
        pt_guess = float(prior_pt)
    elif disc_jd is not None:
        pt_guess = float(disc_jd)

    flux_delta = 1.0
    if np.isfinite(pf_guess) and pf_guess > 0:
        pm = flux_to_mag_scalar(pf_guess)
        if np.isfinite(pm):
            flux_hi = mag_to_flux(pm - 1.0)
            flux_delta = max(flux_hi - pf_guess, pf_guess * 0.5, 0.001)

    # sBV bounds from catalog if available
    sBV_lo, sBV_hi = 0.10, 1.15
    if catalog and 'Ia_norm' in catalog:
        sv = catalog['Ia_norm'].get('s_values', [0.10, 1.15])
        sBV_lo = float(np.min(sv)); sBV_hi = float(np.max(sv))

    prior_sBV = obj.get("prior_sBV")
    sBV_guess = float(prior_sBV) if (prior_sBV is not None and np.isfinite(float(prior_sBV))) else 0.5

    return {
        "peak_flux_guess": pf_guess,
        "peak_time_guess": pt_guess,
        "sBV_guess":       sBV_guess,
        "flux_delta":      flux_delta,
        "time_delta":      50.0,
        "sBV_lo":          sBV_lo,
        "sBV_hi":          sBV_hi,
        "disc_jd":         disc_jd,
    }


# ── Viewer trace builder ──────────────────────────────────────────────────────

BAND_COLOR = {"g": "#10b981", "V": "#f59e0b"}
BAND_LABEL = {"g": "g-band",  "V": "V-band"}

def build_plotly_traces(obj, show_flux=False, x_mode='jd',
                        jd_min=None, jd_max=None) -> dict:
    traces = []; has_data = False
    yaxis_title = "Flux (mJy)" if show_flux else "Magnitude"
    xaxis_title = "UT Date"   if x_mode == 'ut' else "Julian Date (JD)"
    all_dfs = {}; unique_cameras = set()

    for band in ("g", "V"):
        fpath = obj.get(f"{band}_file")
        if not fpath or not os.path.isfile(fpath): continue
        df = parse_lc_file(fpath, band_hint=band)
        if df is not None and not df.empty:
            all_dfs[band] = df
            unique_cameras.update(df["camera"].unique())

    unique_cameras = sorted(unique_cameras)
    syms = ['circle','square','diamond','cross','x','triangle-up','star','hexagram']
    cam_sym = {c: syms[i % len(syms)] for i, c in enumerate(unique_cameras)}
    seen_cams = set()
    y_min, y_max = float('inf'), float('-inf')

    for band in ("g", "V"):
        df = all_dfs.get(band)
        if df is None or df.empty: continue
        color = BAND_COLOR[band]; label = BAND_LABEL[band]

        if show_flux:
            det_all  = df[df["flux"].notna() & np.isfinite(df["flux"].values) & (df["flux"] != 99.990)]
            ulim_all = pd.DataFrame()
        else:
            vm = (df["mag"] > 0) & (df["mag"] < 50)
            det_all  = df[~df["upper_limit"] & vm]
            ulim_all = df[ df["upper_limit"] & vm]

        if not det_all.empty or not ulim_all.empty:
            traces.append({"type":"scatter","x":[],"y":[],"mode":"markers",
                           "name":label,"legendgroup":f"dummy_{label}",
                           "marker":{"color":color,"symbol":"circle","size":14},
                           "legend":"legend","showlegend":True,"hoverinfo":"skip"})

        for cam in df["camera"].unique():
            det  = det_all[det_all["camera"]==cam]   if not det_all.empty  else pd.DataFrame()
            ulim = ulim_all[ulim_all["camera"]==cam] if not ulim_all.empty else pd.DataFrame()
            show_leg = cam not in seen_cams
            if show_leg: seen_cams.add(cam)

            if not det.empty:
                if jd_min is not None and jd_max is not None:
                    vis = det[(det["JD"] >= jd_min) & (det["JD"] <= jd_max)]
                    if not vis.empty:
                        vy  = (vis["flux"] - vis["flux_err"]).min() if show_flux else (vis["mag"] - vis["mag_err"]).min()
                        vy2 = (vis["flux"] + vis["flux_err"]).max() if show_flux else (vis["mag"] + vis["mag_err"]).max()
                        if vy < y_min: y_min = vy
                        if vy2 > y_max: y_max = vy2
                x_v = det["UT_date"].tolist() if x_mode=='ut' else det["JD"].tolist()
                xl  = "UT" if x_mode == 'ut' else "JD"
                if show_flux:
                    y_v = det["flux"].tolist(); y_e = det["flux_err"].clip(lower=0).tolist()
                    cd  = list(zip(det["flux_err"].tolist(), det["IMAGE"].tolist()))
                    hov = f"<b>{label}</b><br>{xl}: %{{x}}<br>Flux: %{{y:.4f}} ± %{{customdata[0]:.4f}} mJy<br>Image: %{{customdata[1]}}<extra></extra>"
                else:
                    y_v = det["mag"].tolist(); y_e = det["mag_err"].clip(0, 5).tolist()
                    cd  = list(zip(det["mag_err"].tolist(), det["IMAGE"].tolist()))
                    hov = f"<b>{label}</b><br>{xl}: %{{x}}<br>Mag: %{{y:.3f}} ± %{{customdata[0]:.3f}}<br>Image: %{{customdata[1]}}<extra></extra>"
                traces.append({
                    "type":"scatter","x":x_v,"y":y_v,
                    "error_y":{"type":"data","array":y_e,"visible":True,"thickness":1.5,"width":5,"color":color},
                    "mode":"markers","name":cam,"legendgroup":cam,"legend":"legend2",
                    "showlegend":show_leg,
                    "marker":{"color":color,"symbol":cam_sym[cam],"size":14,"opacity":0.95},
                    "hovertemplate":hov,"customdata":cd,
                })
                has_data = True; show_leg = False

            if not ulim.empty:
                if not show_flux and jd_min is not None and jd_max is not None:
                    vis = ulim[(ulim["JD"]>=jd_min)&(ulim["JD"]<=jd_max)]
                    if not vis.empty:
                        if vis["mag"].min() < y_min: y_min = vis["mag"].min()
                        if vis["mag"].max() > y_max: y_max = vis["mag"].max()
                x_v = ulim["UT_date"].tolist() if x_mode=='ut' else ulim["JD"].tolist()
                xl  = "UT" if x_mode == 'ut' else "JD"
                traces.append({
                    "type":"scatter","x":x_v,"y":ulim["mag"].tolist(),
                    "mode":"markers","name":cam,"legendgroup":cam,"legend":"legend2",
                    "showlegend":show_leg,
                    "marker":{"symbol":"triangle-down","color":color,"size":14,"opacity":0.4},
                    "hovertemplate":f"<b>{label} lim.</b><br>{xl}: %{{x}}<br>Mag limit: %{{y:.3f}}<br>Image: %{{customdata[0]}}<extra></extra>",
                    "customdata":list(zip(ulim["IMAGE"].tolist())),
                })

    x_range = y_range = None
    if jd_min is not None and jd_max is not None:
        if x_mode == 'ut':
            x_range = [pd.to_datetime(jd_min, unit='D', origin='julian').strftime('%Y-%m-%d %H:%M:%S.%f'),
                       pd.to_datetime(jd_max, unit='D', origin='julian').strftime('%Y-%m-%d %H:%M:%S.%f')]
        else:
            x_range = [jd_min, jd_max]
        if y_min != float('inf') and y_max != float('-inf'):
            rng = y_max - y_min or 1.0
            y_range = [y_min - 0.05*rng, y_max + 0.05*rng]
            if not show_flux: y_range = [y_range[1], y_range[0]]

    return {"traces":traces,"yaxis_title":yaxis_title,"xaxis_title":xaxis_title,
            "has_data":has_data,"x_range":x_range,"y_range":y_range}


def name_from_file(filepath: str) -> str:
    stem = Path(filepath).stem
    n = re.sub(r'_ASASSN_[gV]$', '', stem, flags=re.IGNORECASE)
    return n if n else stem

def band_from_file(filepath: str) -> Optional[str]:
    stem = Path(filepath).stem.lower()
    if stem.endswith('_asassn_g'): return 'g'
    if stem.endswith('_asassn_v'): return 'V'
    return None

def detect_id_column(df: pd.DataFrame) -> Optional[str]:
    preferred = ['TNS_Name_clean','SNName','my_index','myindex','my_id','myid',
                 'id','name','object','objname','obj_name','target','designation']
    cols_lower = {c.lower().lstrip('#').strip(): c for c in df.columns}
    for p in preferred:
        if p.lower() in cols_lower: return cols_lower[p.lower()]
    for col in df.columns:
        s = df[col].dropna()
        if s.dtype == object and pd.to_numeric(s, errors='coerce').isna().sum() / max(len(s),1) > 0.5:
            return col
    return df.columns[0] if len(df.columns) > 0 else None


@app.route("/api/defaults")
def api_defaults():
    return jsonify(DEFAULTS)


# ═══════════════════════════════════════════════════════════════════════════════
# GENERAL / SHARED ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/logo/<theme>")
def serve_logo(theme):
    t = "light" if theme == "light" else "dark"
    for ext in ("jpg","jpeg","png"):
        c = SCRIPT_DIR / f"ASASSN_{t}.{ext}"
        if c.is_file(): return send_file(str(c))
    abort(404)

@app.route("/api/plot")
def api_plot():
    n = len(OBJECTS)
    if n == 0: return jsonify({"error":"no objects loaded"}), 400
    idx = max(0, min(int(request.args.get("index", STATE["current"])), n-1))
    STATE["current"] = idx
    obj = OBJECTS[idx]
    show_flux = request.args.get("flux","0") == "1"
    x_mode    = request.args.get("xaxis","jd")
    prev_mode = request.args.get("prev_xmode","")
    xmin_str  = request.args.get("xmin",""); xmax_str = request.args.get("xmax","")
    jd_min = jd_max = None
    if xmin_str and xmax_str and prev_mode:
        try:
            if prev_mode == 'ut':
                jd_min = pd.to_datetime(xmin_str).to_julian_date()
                jd_max = pd.to_datetime(xmax_str).to_julian_date()
            else:
                jd_min = float(xmin_str); jd_max = float(xmax_str)
        except Exception: pass
    res = build_plotly_traces(obj, show_flux=show_flux, x_mode=x_mode, jd_min=jd_min, jd_max=jd_max)
    return jsonify({**res,"name":obj["name"],"index":idx,"total":n,
                    "has_g":bool(obj.get("g_file") and os.path.isfile(obj.get("g_file",""))),
                    "has_V":bool(obj.get("V_file") and os.path.isfile(obj.get("V_file","")))})

@app.route("/api/upload_files", methods=["POST"])
def api_upload_files():
    files = request.files.getlist("files")
    if not files: return jsonify({"error":"no files received"}), 400
    upload_dir = Path("/tmp/asassn_uploads"); upload_dir.mkdir(exist_ok=True)
    curr_name = OBJECTS[STATE["current"]]["name"] if OBJECTS else None
    obj_map = {o["name"]: o for o in OBJECTS}
    for f in files:
        if not f.filename: continue
        dest = upload_dir / f.filename; f.save(str(dest))
        name = name_from_file(f.filename); band = band_from_file(f.filename)
        if name not in obj_map:
            obj_map[name] = _make_object(name)
        o = obj_map[name]
        if band == "g":
            o["g_file"] = str(dest)
            if o["band"] == "g": o["band"] = "g"
        elif band == "V":
            o["V_file"] = str(dest)
            if not o.get("g_file"): o["band"] = "V"
        else:
            o["g_file"] = str(dest)
    OBJECTS.clear()
    OBJECTS.extend(sorted(obj_map.values(), key=lambda x: x["name"]))
    if curr_name:
        try: STATE["current"] = next(i for i,o in enumerate(OBJECTS) if o["name"]==curr_name)
        except StopIteration: STATE["current"] = 0
    else: STATE["current"] = 0
    return jsonify({"loaded":len(OBJECTS),"names":[o["name"] for o in OBJECTS]})

@app.route("/api/clear", methods=["POST"])
def api_clear():
    OBJECTS.clear(); STATE["current"] = 0
    FIT_STATE["results"].clear(); FIT_STATE["completed"].clear()
    return jsonify({"status":"ok"})

@app.route("/api/remove", methods=["POST"])
def api_remove():
    data = request.json or {}; name = data.get("name")
    curr_name = OBJECTS[STATE["current"]]["name"] if OBJECTS else None
    remaining = [o for o in OBJECTS if o["name"] != name]
    OBJECTS.clear(); OBJECTS.extend(remaining)
    FIT_STATE["results"].pop(name, None); FIT_STATE["completed"].discard(name)
    if not OBJECTS: STATE["current"] = 0
    else:
        try: STATE["current"] = next(i for i,o in enumerate(OBJECTS) if o["name"]==curr_name)
        except StopIteration: STATE["current"] = min(STATE["current"], len(OBJECTS)-1)
    return jsonify({"status":"ok"})

@app.route("/api/objects")
def api_objects():
    return jsonify({"objects":[{"name":o["name"],
                                 "has_g":bool(o.get("g_file") and os.path.isfile(o["g_file"])),
                                 "has_V":bool(o.get("V_file") and os.path.isfile(o["V_file"]))}
                                for o in OBJECTS], "current":STATE["current"]})

@app.route("/api/jump_to_name", methods=["POST"])
def api_jump_to_name():
    data = request.json or {}; name = data.get("name","").strip().lower()
    for i, obj in enumerate(OBJECTS):
        if obj["name"].lower() == name:
            STATE["current"] = i; return jsonify({"index":i,"found":True})
    for i, obj in enumerate(OBJECTS):
        if name in obj["name"].lower():
            STATE["current"] = i; return jsonify({"index":i,"found":True,"matched":obj["name"]})
    return jsonify({"found":False})


# ═══════════════════════════════════════════════════════════════════════════════
# FITTER ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

def _fit_obj_serialise(obj) -> dict:
    name = obj["name"]
    res  = FIT_STATE["results"].get(name, {})
    return {
        "name":           name,
        "has_g":          bool(obj.get("g_file") and os.path.isfile(obj.get("g_file",""))),
        "has_V":          bool(obj.get("V_file") and os.path.isfile(obj.get("V_file",""))),
        "band":           obj.get("band","g"),
        "active_bands":   obj.get("active_bands","primary"),
        "fitted":         bool(res),
        "completed":      name in FIT_STATE["completed"],
        "type_to_use":    obj.get("type_to_use",""),
        "redshift":       _safe(obj.get("redshift")),
        "flag":           obj.get("flag"),
        "sub_type":       res.get("sub_type",""),
        "has_undo":       len(obj.get("undo_stack",[])) > 0,
    }


@app.route("/api/fit/set_template_dir", methods=["POST"])
def api_fit_set_template_dir():
    data = request.json or {}
    tdir = data.get("template_dir","").strip()
    if not tdir or not os.path.isdir(tdir):
        return jsonify({"error":f"Directory not found: {tdir}"}), 400
    FIT_STATE["template_dir"]     = tdir
    FIT_STATE["template_catalog"] = None   # force reload
    catalog = _get_catalog()
    available = sorted(catalog.keys()) if catalog else []
    return jsonify({"status":"ok","available_templates":available,"loaded":len(available)})


@app.route("/api/fit/available_templates")
def api_fit_available_templates():
    catalog = _get_catalog()
    if catalog is None:
        return jsonify({"templates":[],"template_dir":FIT_STATE["template_dir"]})
    out = []
    for st, info in catalog.items():
        out.append({"subtype":st,"type":info["type"]})
    return jsonify({"templates":sorted(out, key=lambda x: x["subtype"]),
                    "template_dir":FIT_STATE["template_dir"]})


@app.route("/api/fit/load_master_csv", methods=["POST"])
def api_fit_load_master_csv():
    csv_file     = request.files.get("csv_file")
    lc_dir       = request.form.get("lc_dir","").strip()
    output_csv   = request.form.get("output_csv","").strip()

    if not csv_file: return jsonify({"error":"No CSV file uploaded"}), 400
    try:
        raw = csv_file.read().decode("utf-8", errors="replace")
        lines = raw.split("\n")
        if lines and lines[0].startswith("#"): lines[0] = lines[0].lstrip("#")
        df = pd.read_csv(io.StringIO("\n".join(lines)))
        df.columns = [c.strip().lstrip("#").strip() for c in df.columns]
    except Exception as e: return jsonify({"error":f"CSV parse error: {e}"}), 400

    # Determine name column — prefer the standard catalog columns, otherwise
    # fall back to the same fuzzy detection used for a plain/arbitrary CSV.
    name_col = None
    for nc in ("TNS_Name_clean","SNName","Name","name"):
        if nc in df.columns:
            name_col = nc; break
    if name_col is None:
        name_col = detect_id_column(df)
    if name_col is None:
        return jsonify({"error":"Could not detect a name/ID column in this CSV."}), 400

    if output_csv:
        FIT_STATE["output_csv"] = output_csv

    new_objects, missing = [], []
    for _, row in df.iterrows():
        # Name
        raw_name = str(row.get(name_col,"")).strip()
        if not raw_name or raw_name.lower() in ("nan",""):
            # Try fallback name column
            for nc2 in ("TNS_Name_clean","SNName","Name","name"):
                if nc2 != name_col and nc2 in row.index:
                    candidate = str(row.get(nc2,"")).strip()
                    if candidate and candidate.lower() not in ("nan",""):
                        raw_name = candidate; break
        if not raw_name or raw_name.lower() in ("nan",""): continue

        # Redshift
        try: redshift = float(row.get("redshift_to_use", float('nan')))
        except: redshift = float('nan')

        # Distance modulus: prefer dist_mod_z_indep, then dist_mod_z, then compute
        dist_mod = float('nan')
        for dc in ("dist_mod_z_indep","dist_mod_z"):
            if dc in row.index:
                v = row.get(dc)
                try:
                    fv = float(v)
                    if np.isfinite(fv): dist_mod = fv; break
                except: pass
        if not np.isfinite(dist_mod):
            dist_mod = compute_dist_mod(redshift)

        # K-correction
        for kc in ("K_corr","Kcorr","kcorr","k_corr"):
            if kc in row.index:
                try: k_corr = float(row[kc]); break
                except: k_corr = 0.0
        else: k_corr = 0.0

        # Extinction
        for ec in ("gal_ext","extinction_Ag","Ag","E_BV"):
            if ec in row.index:
                try: gal_ext = float(row[ec]); break
                except: gal_ext = 0.0
        else: gal_ext = 0.0

        # Discovery date
        disc_jd = None
        for dc in ("Discovery Date (UT)","disc_date","discovery_date"):
            if dc in row.index:
                disc_jd = ut_to_jd(str(row[dc]).strip()); break

        # Band from sample column
        sample = str(row.get("sample","g")).strip().upper()
        band   = "V" if sample == "V" else "g"

        # Type
        type_to_use = str(row.get("type_to_use","")).strip()

        # Prior fit info
        prior_sub   = str(row.get("fit_sub_type","")).strip() or None
        prior_pt    = _nan_or(row.get("fit_peak_time")) if "fit_peak_time"  in row.index else None
        prior_amag  = _nan_or(row.get("fit_app_mag"))   if "fit_app_mag"    in row.index else None
        prior_sBV   = _nan_or(row.get("fit_sBV"))       if "fit_sBV"        in row.index else None

        # LC files
        g_file = V_file = None
        if lc_dir:
            gc = os.path.join(lc_dir, f"{raw_name}_ASASSN_g.dat")
            Vc = os.path.join(lc_dir, f"{raw_name}_ASASSN_V.dat")
            if os.path.isfile(gc): g_file = gc
            if os.path.isfile(Vc): V_file = Vc
            if not g_file and not V_file: missing.append(raw_name)

        # Preserve identifying meta columns for output
        id_cols = ["TNS_Name_clean","SNName","Name","RA_deg","DEC_deg","ra_hms","dec_dms",
                   "redshift_to_use","type_to_use","Discovery Date (UT)","sample"]
        meta = {c: row[c] for c in id_cols if c in row.index}

        # Re-use existing object if already in OBJECTS
        existing = _find_obj(raw_name)
        if existing:
            if g_file:  existing["g_file"]  = g_file
            if V_file:  existing["V_file"]   = V_file
            existing["band"]           = band
            existing["redshift"]       = redshift if np.isfinite(redshift) else None
            existing["dist_mod"]       = dist_mod if np.isfinite(dist_mod) else None
            existing["k_corr"]         = k_corr
            existing["gal_ext"]        = gal_ext
            existing["disc_jd"]        = disc_jd
            existing["type_to_use"]    = type_to_use
            existing["prior_sub_type"] = prior_sub
            existing["prior_peak_time"]= prior_pt
            existing["prior_app_mag"]  = prior_amag
            existing["prior_sBV"]      = prior_sBV
            existing["meta"]           = meta
            new_objects.append(existing)
        else:
            o = _make_object(raw_name, g_file=g_file, V_file=V_file, band=band, meta=meta)
            o["redshift"]       = redshift if np.isfinite(redshift) else None
            o["dist_mod"]       = dist_mod if np.isfinite(dist_mod) else None
            o["k_corr"]         = k_corr
            o["gal_ext"]        = gal_ext
            o["disc_jd"]        = disc_jd
            o["type_to_use"]    = type_to_use
            o["prior_sub_type"] = prior_sub
            o["prior_peak_time"]= prior_pt
            o["prior_app_mag"]  = prior_amag
            o["prior_sBV"]      = prior_sBV
            new_objects.append(o)

    if not new_objects: return jsonify({"error":"No valid objects found"}), 400

    # Merge: keep objects not in new list (from viewer drops), prepend new
    old_names = {o["name"] for o in new_objects}
    kept_old  = [o for o in OBJECTS if o["name"] not in old_names]
    OBJECTS.clear()
    OBJECTS.extend(new_objects + kept_old)
    OBJECTS.sort(key=lambda x: x["name"])
    STATE["current"] = 0; FIT_STATE["current"] = 0

    # Restore saved progress from output CSV
    if output_csv and os.path.isfile(output_csv):
        try:
            out_df = pd.read_csv(output_csv)
            for _, row in out_df.iterrows():
                n2 = str(row.get("Name","")).strip()
                o  = _find_obj(n2)
                if o and pd.notna(row.get("peak_flux")):
                    FIT_STATE["results"][n2] = {k: row.get(k) for k in row.index
                                                if k not in ("Name","Removed_JDs")}
                    rem = row.get("Removed_JDs","[]")
                    try:
                        rjds = json.loads(rem)
                        if isinstance(rjds, list): o["removed_jds"] = rjds
                    except: pass
                if pd.notna(row.get("flag")):
                    o = _find_obj(n2)
                    if o: o["flag"] = row["flag"]
                is_done = str(row.get("Completed","")).strip().lower() in ("true","1","t","yes")
                if is_done: FIT_STATE["completed"].add(n2)
        except Exception as e:
            print(f"Notice: could not restore progress: {e}")

    return jsonify({"loaded":len(new_objects),"missing":missing[:20],
                    "names":[o["name"] for o in new_objects[:100]]})


@app.route("/api/fit/objects")
def api_fit_objects():
    return jsonify({"objects":[_fit_obj_serialise(o) for o in OBJECTS],
                    "current":FIT_STATE["current"]})


@app.route("/api/fit/get_guesses")
def api_fit_get_guesses():
    n = len(OBJECTS)
    if n == 0: return jsonify({"error":"no objects loaded"}), 400
    idx = max(0, min(int(request.args.get("index", FIT_STATE["current"])), n-1))
    FIT_STATE["current"] = idx
    obj  = OBJECTS[idx]
    name = obj["name"]

    def _fa(arg_name):
        try: v = request.args.get(arg_name); return float(v) if v not in (None,"","null") else None
        except: return None

    jd_min_fit  = _fa("jd_min_fit")
    jd_max_fit  = _fa("jd_max_fit")
    catalog     = _get_catalog()
    g           = auto_guesses(obj, jd_min=jd_min_fit, jd_max=jd_max_fit, catalog=catalog)
    existing    = FIT_STATE["results"].get(name)
    # strip MC samples from existing result to keep response small
    existing_serial = {k:v for k,v in existing.items()
                       if not k.startswith("mc_")} if existing else None

    return jsonify({
        "name":             name, "index":idx, "total":n,
        "band":             obj.get("band","g"),
        "active_bands":     obj.get("active_bands","primary"),
        "type_to_use":      obj.get("type_to_use",""),
        "redshift":         _safe(obj.get("redshift")),
        "dist_mod":         _safe(obj.get("dist_mod")),
        "k_corr":           _safe(obj.get("k_corr",0.0)),
        "gal_ext":          _safe(obj.get("gal_ext",0.0)),
        "disc_jd":          _safe(g["disc_jd"]),
        "peak_flux_guess":  _safe(g["peak_flux_guess"]),
        "peak_time_guess":  _safe(g["peak_time_guess"]),
        "sBV_guess":        _safe(g["sBV_guess"]),
        "flux_delta":       _safe(g["flux_delta"]),
        "time_delta":       _safe(g["time_delta"]),
        "sBV_lo":           _safe(g["sBV_lo"]),
        "sBV_hi":           _safe(g["sBV_hi"]),
        "prior_sub_type":   obj.get("prior_sub_type",""),
        "has_g":            bool(obj.get("g_file") and os.path.isfile(obj.get("g_file",""))),
        "has_V":            bool(obj.get("V_file") and os.path.isfile(obj.get("V_file",""))),
        "fit_result":       existing_serial,
        "has_mc":           bool(existing and existing.get("mc_peak_flux")),
        "completed":        name in FIT_STATE["completed"],
        "flag":             obj.get("flag"),
        "has_undo":         len(obj.get("undo_stack",[])) > 0,
        "removed_jds":      obj.get("removed_jds",[]),
        "v_scale":          obj.get("v_scale",1.0),
        "error_inflate":    obj.get("error_inflate",0.0),
        "baseline_shifts":  obj.get("baseline_shifts",{}),
    })


def _parse_fit_params(data):
    """Extract and validate common fitting params from request body."""
    def _fv(k):
        v = data.get(k)
        return float(v) if v not in (None,"","null") else None

    pf   = _fv("peak_flux_guess")
    pt   = _fv("peak_time_guess")
    fd   = _fv("flux_delta")
    td   = _fv("time_delta")
    sBV  = _fv("sBV_guess")
    sBVl = _fv("sBV_lo")
    sBVh = _fv("sBV_hi")
    jdmn = _fv("jd_min_fit")
    jdmx = _fv("jd_max_fit")

    if None in (pf, pt, fd, td):
        raise ValueError("peak_flux_guess, peak_time_guess, flux_delta, time_delta are required")

    return pf, pt, fd, td, sBV or 0.5, sBVl or 0.10, sBVh or 1.15, jdmn, jdmx


@app.route("/api/fit/run", methods=["POST"])
def api_fit_run():
    data = request.json or {}
    n = len(OBJECTS)
    if n == 0: return jsonify({"error":"no objects loaded"}), 400
    idx = max(0, min(int(data.get("index", FIT_STATE["current"])), n-1))
    FIT_STATE["current"] = idx
    obj  = OBJECTS[idx]; name = obj["name"]

    try:
        pf, pt, fd, td, sBV_g, sBV_lo, sBV_hi, jd_min_fit, jd_max_fit = _parse_fit_params(data)
    except ValueError as e:
        return jsonify({"error":str(e)}), 400

    sub_type     = (data.get("sub_type") or "Ia_norm").strip()
    use_clipping = bool(data.get("use_clipping", True))
    catalog      = _get_catalog()

    if catalog is None:
        return jsonify({"error":"Template catalog not loaded — set template directory first."}), 400
    if sub_type not in catalog:
        return jsonify({"error":f"Template '{sub_type}' not found. Available: {sorted(catalog.keys())}"}), 400

    tinfo = catalog[sub_type]
    band  = obj.get("band","g")

    _, _, valid = get_valid_fit_data(obj, jd_min_fit, jd_max_fit)
    if valid is None or valid.empty:
        return jsonify({"error":f"No valid data in fitting range for {name}"}), 400

    bounds  = _default_bounds(pf, pt, fd, td, tinfo, sBV_lo, sBV_hi)
    eps     = 1e-6
    p0_clip = np.clip([pf, pt], [bounds[0][0]+eps, bounds[1][0]+eps],
                                [bounds[0][1]-eps, bounds[1][1]-eps]).tolist()
    if tinfo['type'] == '2D':
        p0_clip.append(float(np.clip(sBV_g, sBV_lo+eps, sBV_hi-eps)))
    elif tinfo['type'] == 'CV':
        p0_clip.append(1.0)

    redshift = obj.get("redshift") or 0.0
    times    = valid["JD"].values
    fluxes   = valid["flux"].values
    errors   = valid["flux_err"].values

    result, mask = fit_with_clipping(times, fluxes, errors, redshift,
                                     tinfo, band, p0_clip, bounds, use_clipping)
    if result is None or not result.success:
        return jsonify({"error":f"Fit did not converge for {name}. Try adjusting bounds or clipping."}), 400

    p_best = result.x
    n_pts  = int(mask.sum())
    dof    = max(n_pts - len(p_best), 1)
    chi2_red = float(result.fun) / dof

    # Derived quantities
    app_mag    = flux_to_mag_scalar(p_best[0])
    dist_mod   = obj.get("dist_mod") or compute_dist_mod(redshift)
    k_corr     = obj.get("k_corr", 0.0)
    gal_ext    = obj.get("gal_ext", 0.0)
    abs_mag    = compute_abs_mag(app_mag, dist_mod, k_corr, gal_ext)

    abs_mag_sBV = float('nan')
    if tinfo['type'] == '2D' and len(p_best) > 2:
        sBV_fit = float(p_best[2])
        sv = tinfo.get('s_values', []); Mv = tinfo.get('M_values', [])
        if len(sv) > 0:
            abs_mag_sBV = float(np.interp(sBV_fit, sv, Mv))

    # Boundary warnings
    warnings = []
    EDGE = 1e-3
    def _near(val, lo, hi):
        sp = hi-lo
        if sp <= 0: return True
        return (val-lo)/sp < EDGE or (hi-val)/sp < EDGE

    if _near(p_best[0], bounds[0][0], bounds[0][1]):
        warnings.append(f"Peak flux ({p_best[0]:.4f} mJy) hit the ±Δ boundary — consider widening Δ flux.")
    if _near(p_best[1], bounds[1][0], bounds[1][1]):
        warnings.append(f"Peak time (JD {p_best[1]:.2f}) hit the ±Δ boundary — consider widening Δ time.")

    fit_result = {
        "sub_type":        sub_type,
        "peak_flux":       round(float(p_best[0]),6),
        "peak_time":       round(float(p_best[1]),4),
        "sBV":             round(float(p_best[2]),4) if tinfo['type']=='2D' and len(p_best)>2 else None,
        "stretch":         round(float(p_best[2]),4) if tinfo['type']=='CV' and len(p_best)>2 else None,
        "app_mag":         _safe4(app_mag),
        "abs_mag":         _safe4(abs_mag),
        "abs_mag_sBV":     _safe4(abs_mag_sBV),
        "chi2_red":        round(chi2_red,4),
        "n_pts":           n_pts,
        "warnings":        warnings,
        "disc_jd":         obj.get("disc_jd"),
        "jd_min_fit":      jd_min_fit,
        "jd_max_fit":      jd_max_fit,
        # store bounds/guesses for MC to re-use
        "_p_best":         p_best.tolist(),
        "_bounds":         bounds,
        "_band":           band,
        "_redshift":       float(redshift),
    }
    # placeholders for MC errors
    for k in ("peak_flux_err_low","peak_flux_err_up","peak_time_err_low","peak_time_err_up",
              "sBV_err_low","sBV_err_up","app_mag_err_low","app_mag_err_up",
              "abs_mag_err_low","abs_mag_err_up"):
        fit_result[k] = None

    FIT_STATE["results"][name] = fit_result
    serial = {k:v for k,v in fit_result.items() if not k.startswith("_")}
    return jsonify({**serial, "name":name, "index":idx, "total":n})


@app.route("/api/fit/run_mc", methods=["POST"])
def api_fit_run_mc():
    data = request.json or {}
    n = len(OBJECTS)
    if n == 0: return jsonify({"error":"no objects loaded"}), 400
    idx = max(0, min(int(data.get("index", FIT_STATE["current"])), n-1))
    FIT_STATE["current"] = idx
    obj  = OBJECTS[idx]; name = obj["name"]

    existing = FIT_STATE["results"].get(name)
    if not existing:
        return jsonify({"error":"Run fit first before running MC."}), 400

    n_mc     = int(data.get("n_mc", FIT_STATE["n_mc"]))
    catalog  = _get_catalog()
    if catalog is None: return jsonify({"error":"Template catalog not loaded."}), 400

    sub_type = existing.get("sub_type","Ia_norm")
    if sub_type not in catalog: return jsonify({"error":f"Template '{sub_type}' not found."}), 400

    tinfo    = catalog[sub_type]
    p_best   = existing.get("_p_best")
    bounds   = existing.get("_bounds")
    band     = existing.get("_band", obj.get("band","g"))
    redshift = existing.get("_redshift", float(obj.get("redshift") or 0.0))

    if p_best is None or bounds is None:
        return jsonify({"error":"Fit metadata missing — please re-run the fit."}), 400

    jd_min_fit = existing.get("jd_min_fit"); jd_max_fit = existing.get("jd_max_fit")
    _, _, valid = get_valid_fit_data(obj, jd_min_fit, jd_max_fit)
    if valid is None or valid.empty: return jsonify({"error":"No valid data for MC."}), 400

    times  = valid["JD"].values; fluxes = valid["flux"].values; errors = valid["flux_err"].values

    samples = run_monte_carlo(times, fluxes, errors, redshift, tinfo, band,
                              p_best, bounds, n_mc=n_mc)
    if samples is None or len(samples) == 0:
        return jsonify({"error":"MC produced no valid fits. Try loosening bounds."}), 400

    # Compute derived quantities per sample
    dist_mod = obj.get("dist_mod") or compute_dist_mod(redshift)
    k_corr   = obj.get("k_corr", 0.0); gal_ext = obj.get("gal_ext", 0.0)
    mc_app   = np.array([flux_to_mag_scalar(s[0]) for s in samples])
    mc_abs   = np.array([compute_abs_mag(am, dist_mod, k_corr, gal_ext) for am in mc_app])

    mc_sBV        = None; mc_abs_sBV = None
    if tinfo['type'] == '2D' and samples.shape[1] > 2:
        mc_sBV = samples[:,2].tolist()
        sv = tinfo.get('s_values',[]); Mv = tinfo.get('M_values',[])
        if len(sv):
            mc_abs_sBV = [float(np.interp(s, sv, Mv)) for s in samples[:,2]]

    p_best_arr = np.array(p_best)
    p16 = np.nanpercentile(samples, 16, axis=0)
    p84 = np.nanpercentile(samples, 84, axis=0)
    p16_app = np.nanpercentile(mc_app, 16); p84_app = np.nanpercentile(mc_app, 84)
    p16_abs = np.nanpercentile(mc_abs, 16); p84_abs = np.nanpercentile(mc_abs, 84)

    existing["peak_flux_err_low"] = round(float(p_best_arr[0] - p16[0]),6)
    existing["peak_flux_err_up"]  = round(float(p84[0] - p_best_arr[0]),6)
    existing["peak_time_err_low"] = round(float(p_best_arr[1] - p16[1]),4)
    existing["peak_time_err_up"]  = round(float(p84[1] - p_best_arr[1]),4)
    if mc_sBV is not None:
        existing["sBV_err_low"] = round(float(p_best_arr[2] - p16[2]),4)
        existing["sBV_err_up"]  = round(float(p84[2] - p_best_arr[2]),4)
    existing["app_mag_err_low"] = round(float(float(existing.get("app_mag",0) or 0) - p16_app),4)
    existing["app_mag_err_up"]  = round(float(p84_app - float(existing.get("app_mag",0) or 0)),4)
    existing["abs_mag_err_low"] = round(float(float(existing.get("abs_mag",0) or 0) - p16_abs),4) if np.isfinite(p16_abs) else None
    existing["abs_mag_err_up"]  = round(float(p84_abs - float(existing.get("abs_mag",0) or 0)),4) if np.isfinite(p84_abs) else None
    if mc_abs_sBV is not None:
        mc_abs_sBV_arr = np.array(mc_abs_sBV)
        p16_abs_sBV = np.nanpercentile(mc_abs_sBV_arr, 16)
        p84_abs_sBV = np.nanpercentile(mc_abs_sBV_arr, 84)
        best_abs_sBV = float(existing.get("abs_mag_sBV") or 0)
        existing["abs_mag_sBV_err_low"] = round(float(best_abs_sBV - p16_abs_sBV), 4) if np.isfinite(p16_abs_sBV) else None
        existing["abs_mag_sBV_err_up"]  = round(float(p84_abs_sBV - best_abs_sBV), 4) if np.isfinite(p84_abs_sBV) else None

    # Store samples (for client-side histogram rendering)
    existing["mc_peak_flux"] = samples[:,0].tolist()
    existing["mc_peak_time"] = samples[:,1].tolist()
    existing["mc_sBV"]       = mc_sBV
    existing["mc_app_mag"]   = mc_app.tolist()
    existing["mc_abs_mag"]   = mc_abs.tolist() if np.any(np.isfinite(mc_abs)) else None
    existing["mc_abs_mag_sBV"] = mc_abs_sBV

    serial = {k:v for k,v in existing.items() if not k.startswith("_")}
    return jsonify({**serial, "name":name, "index":idx, "n_mc_success":len(samples)})


@app.route("/api/fit/mc_histograms")
def api_fit_mc_histograms():
    """Return MC sample arrays for histogram rendering (lightweight — no model evaluation)."""
    n = len(OBJECTS)
    if n == 0: return jsonify({"error":"no objects"}), 400
    idx = max(0, min(int(request.args.get("index", FIT_STATE["current"])), n-1))
    obj  = OBJECTS[idx]; name = obj["name"]
    res  = FIT_STATE["results"].get(name)
    if not res or not res.get("mc_peak_flux"):
        return jsonify({"has_mc": False}), 200
    return jsonify({
        "has_mc":      True,
        "name":        name,
        "index":       idx,
        "n_mc_success": len(res["mc_peak_flux"]),
        # sample arrays for histograms
        "mc_peak_flux": res.get("mc_peak_flux"),
        "mc_peak_time": res.get("mc_peak_time"),
        "mc_app_mag":   res.get("mc_app_mag"),
        "mc_abs_mag":     res.get("mc_abs_mag"),
        "mc_sBV":         res.get("mc_sBV"),
        "mc_abs_mag_sBV": res.get("mc_abs_mag_sBV"),
        # best values for the vertical marker lines
        "peak_flux":    res.get("peak_flux"),
        "peak_time":    res.get("peak_time"),
        "app_mag":      res.get("app_mag"),
        "abs_mag":      res.get("abs_mag"),
        "sBV":          res.get("sBV"),
        "abs_mag_sBV":          res.get("abs_mag_sBV"),
        "abs_mag_sBV_err_low":  res.get("abs_mag_sBV_err_low"),
        "abs_mag_sBV_err_up":   res.get("abs_mag_sBV_err_up"),
    })


@app.route("/api/fit/find_best", methods=["POST"])
def api_fit_find_best():
    data = request.json or {}
    n = len(OBJECTS)
    if n == 0: return jsonify({"error":"no objects loaded"}), 400
    idx = max(0, min(int(data.get("index", FIT_STATE["current"])), n-1))
    FIT_STATE["current"] = idx
    obj  = OBJECTS[idx]; name = obj["name"]

    try:
        pf, pt, fd, td, sBV_g, sBV_lo, sBV_hi, jd_min_fit, jd_max_fit = _parse_fit_params(data)
    except ValueError as e:
        return jsonify({"error":str(e)}), 400

    use_clipping = bool(data.get("use_clipping", True))
    catalog = _get_catalog()
    if catalog is None: return jsonify({"error":"Template catalog not loaded."}), 400

    band     = obj.get("band","g"); redshift = obj.get("redshift") or 0.0
    _, _, valid = get_valid_fit_data(obj, jd_min_fit, jd_max_fit)
    if valid is None or valid.empty: return jsonify({"error":"No valid data."}), 400

    times = valid["JD"].values; fluxes = valid["flux"].values; errors = valid["flux_err"].values
    best_chi2 = float('inf'); best_sub = None; best_result = None

    for sub_type, tinfo in catalog.items():
        try:
            bounds = _default_bounds(pf, pt, fd, td, tinfo, sBV_lo, sBV_hi)
            eps    = 1e-6
            p0     = np.clip([pf, pt], [bounds[0][0]+eps, bounds[1][0]+eps],
                                       [bounds[0][1]-eps, bounds[1][1]-eps]).tolist()
            if tinfo['type'] == '2D':   p0.append(float(np.clip(sBV_g, sBV_lo+eps, sBV_hi-eps)))
            elif tinfo['type'] == 'CV': p0.append(1.0)

            result, mask = fit_with_clipping(times, fluxes, errors, redshift,
                                             tinfo, band, p0, bounds, use_clipping)
            if result is None or not result.success: continue

            dof  = max(int(mask.sum()) - len(p0), 1)
            chi2 = float(result.fun) / dof
            if chi2 < best_chi2:
                best_chi2 = chi2; best_sub = sub_type; best_result = result
        except Exception:
            continue

    if best_sub is None:
        return jsonify({"error":"No templates converged. Try adjusting bounds."}), 400

    # Forward the best sub_type to the regular fit endpoint logic inline
    tinfo   = catalog[best_sub]; p_best  = best_result.x
    n_pts   = int(len(times))
    chi2_red= best_chi2

    app_mag = flux_to_mag_scalar(p_best[0])
    dist_mod= obj.get("dist_mod") or compute_dist_mod(redshift)
    abs_mag = compute_abs_mag(app_mag, dist_mod, obj.get("k_corr",0.0), obj.get("gal_ext",0.0))
    abs_mag_sBV = float('nan')
    if tinfo['type']=='2D' and len(p_best)>2:
        sv=tinfo.get('s_values',[]); Mv=tinfo.get('M_values',[])
        if len(sv): abs_mag_sBV = float(np.interp(float(p_best[2]), sv, Mv))

    bounds = _default_bounds(pf, pt, fd, td, tinfo, sBV_lo, sBV_hi)
    fit_result = {
        "sub_type": best_sub,
        "peak_flux": round(float(p_best[0]),6),
        "peak_time": round(float(p_best[1]),4),
        "sBV":      round(float(p_best[2]),4) if tinfo['type']=='2D' and len(p_best)>2 else None,
        "stretch":  round(float(p_best[2]),4) if tinfo['type']=='CV' and len(p_best)>2 else None,
        "app_mag":  _safe4(app_mag),
        "abs_mag":  _safe4(abs_mag),
        "abs_mag_sBV": _safe4(abs_mag_sBV),
        "chi2_red": round(chi2_red,4), "n_pts": n_pts,
        "warnings": [], "disc_jd": obj.get("disc_jd"),
        "jd_min_fit": jd_min_fit, "jd_max_fit": jd_max_fit,
        "_p_best": p_best.tolist(), "_bounds": bounds,
        "_band": band, "_redshift": float(redshift),
    }
    for k in ("peak_flux_err_low","peak_flux_err_up","peak_time_err_low","peak_time_err_up",
              "sBV_err_low","sBV_err_up","app_mag_err_low","app_mag_err_up",
              "abs_mag_err_low","abs_mag_err_up"):
        fit_result[k] = None

    FIT_STATE["results"][name] = fit_result
    serial = {k:v for k,v in fit_result.items() if not k.startswith("_")}
    return jsonify({**serial, "name":name, "index":idx, "total":n,
                    "best_sub_type":best_sub})


@app.route("/api/fit/run_all", methods=["POST"])
def api_fit_run_all():
    data = request.json or {}
    catalog = _get_catalog()
    if catalog is None: return jsonify({"error":"Template catalog not loaded."}), 400

    sub_type     = (data.get("sub_type") or "Ia_norm").strip()
    if sub_type not in catalog:
        return jsonify({"error":f"Template '{sub_type}' not in catalog."}), 400
    use_clipping = bool(data.get("use_clipping", True))

    def _fv(k):
        v = data.get(k); return float(v) if v not in (None,"","null") else None
    jd_min_fit = _fv("jd_min_fit"); jd_max_fit = _fv("jd_max_fit")

    fitted = 0; skipped = 0; results_out = []
    for idx, obj in enumerate(OBJECTS):
        name = obj["name"]
        if name in FIT_STATE["completed"]:
            results_out.append({"name":name,"status":"skipped","reason":"already completed"}); skipped+=1; continue
        has_data_file = bool(
            (obj.get("g_file") and os.path.isfile(obj.get("g_file",""))) or
            (obj.get("V_file") and os.path.isfile(obj.get("V_file","")))
        )
        if not has_data_file:
            results_out.append({"name":name,"status":"skipped","reason":"no LC file"}); skipped+=1; continue

        tinfo    = catalog[sub_type]
        redshift = obj.get("redshift") or 0.0
        _, _, valid = get_valid_fit_data(obj, jd_min_fit, jd_max_fit)
        if valid is None or valid.empty:
            results_out.append({"name":name,"status":"skipped","reason":"no valid data"}); skipped+=1; continue

        g = auto_guesses(obj, jd_min=jd_min_fit, jd_max=jd_max_fit, catalog=catalog)
        pf = g["peak_flux_guess"]; pt = g["peak_time_guess"]
        if not (np.isfinite(pf) and np.isfinite(pt)):
            results_out.append({"name":name,"status":"skipped","reason":"could not auto-guess"}); skipped+=1; continue

        band    = obj.get("band","g")
        bounds  = _default_bounds(pf, pt, g["flux_delta"], g["time_delta"], tinfo,
                                  g["sBV_lo"], g["sBV_hi"])
        eps     = 1e-6
        p0      = np.clip([pf, pt], [bounds[0][0]+eps, bounds[1][0]+eps],
                                    [bounds[0][1]-eps, bounds[1][1]-eps]).tolist()
        if tinfo['type'] == '2D':   p0.append(float(np.clip(g["sBV_guess"], g["sBV_lo"]+eps, g["sBV_hi"]-eps)))
        elif tinfo['type'] == 'CV': p0.append(1.0)

        times = valid["JD"].values; fluxes = valid["flux"].values; errors = valid["flux_err"].values
        try:
            result, mask = fit_with_clipping(times, fluxes, errors, redshift,
                                             tinfo, band, p0, bounds, use_clipping)
            if result is None or not result.success:
                results_out.append({"name":name,"status":"error","reason":"did not converge"}); skipped+=1; continue
        except Exception as e:
            results_out.append({"name":name,"status":"error","reason":str(e)}); skipped+=1; continue

        p_best  = result.x; n_pts = int(mask.sum())
        chi2_red = float(result.fun) / max(n_pts - len(p_best), 1)
        app_mag  = flux_to_mag_scalar(p_best[0])
        dist_mod = obj.get("dist_mod") or compute_dist_mod(redshift)
        abs_mag  = compute_abs_mag(app_mag, dist_mod, obj.get("k_corr",0.0), obj.get("gal_ext",0.0))

        FIT_STATE["results"][name] = {
            "sub_type":  sub_type,
            "peak_flux": round(float(p_best[0]),6),
            "peak_time": round(float(p_best[1]),4),
            "sBV":       round(float(p_best[2]),4) if tinfo['type']=='2D' and len(p_best)>2 else None,
            "app_mag":   _safe4(app_mag), "abs_mag": _safe4(abs_mag),
            "chi2_red":  round(chi2_red,4), "n_pts": n_pts,
            "warnings":  [], "disc_jd": obj.get("disc_jd"),
            "jd_min_fit": jd_min_fit, "jd_max_fit": jd_max_fit,
            "_p_best": p_best.tolist(), "_bounds": bounds,
            "_band": band, "_redshift": float(redshift),
        }
        fitted += 1
        results_out.append({"name":name,"status":"ok","index":idx,
                             "best_flux":round(float(p_best[0]),6),"app_mag":_safe4(app_mag),
                             "chi2_red":round(chi2_red,4),"n_pts":n_pts})

    return jsonify({"fitted":fitted,"skipped":skipped,"results":results_out})


@app.route("/api/fit/plot")
def api_fit_plot():
    n = len(OBJECTS)
    if n == 0: return jsonify({"error":"no objects loaded"}), 400
    idx = max(0, min(int(request.args.get("index", FIT_STATE["current"])), n-1))
    obj  = OBJECTS[idx]; name = obj["name"]
    res  = FIT_STATE["results"].get(name)
    if res is None: return jsonify({"error":"No fit result — run the fit first."}), 400

    show_flux = request.args.get("flux", "1") == "1"
    x_mode    = request.args.get("xaxis", "jd")
    xaxis_title = "UT Date" if x_mode == 'ut' else "Julian Date (JD)"
    yaxis_title = "Flux (mJy)" if show_flux else "Magnitude"

    # ── coordinate helpers ────────────────────────────────────────────────
    def _jd_to_ut(jd_val):
        return pd.to_datetime(float(jd_val), unit='D', origin='julian').strftime('%Y-%m-%d %H:%M:%S.%f')

    def _x(jd_val):
        return _jd_to_ut(jd_val) if x_mode == 'ut' else jd_val

    def _xl(jd_list):
        return [_jd_to_ut(j) for j in jd_list] if x_mode == 'ut' else list(jd_list)

    def _y(flux_val):
        if show_flux:
            return None if (flux_val is None or not np.isfinite(flux_val)) else round(float(flux_val), 6)
        m = flux_to_mag_scalar(flux_val)
        return None if not np.isfinite(m) else round(m, 4)

    def _yl(flux_list):
        return [_y(v) for v in flux_list]

    def _yerr(flux_arr, err_arr):
        if show_flux:
            return err_arr.tolist()
        result = []
        for f, fe in zip(flux_arr, err_arr):
            if np.isfinite(f) and np.isfinite(fe) and f > 0:
                result.append(round(float(FLUX_TO_MAG_CONST * fe / f), 4))
            else:
                result.append(None)
        return result

    fit_band  = obj.get("band","g")
    active    = obj.get("active_bands","primary")
    fit_bands = {fit_band} if active != "both" else {"g","V"}
    BAND_DIM  = {"g": "rgba(16,185,129,0.3)", "V": "rgba(245,158,11,0.3)"}
    best_flux = float(res["peak_flux"]); best_time = float(res["peak_time"])
    redshift  = res.get("_redshift", float(obj.get("redshift") or 0.0))
    disc_jd   = res.get("disc_jd") or obj.get("disc_jd")
    jd_min_fit = res.get("jd_min_fit"); jd_max_fit = res.get("jd_max_fit")
    sub_type  = res.get("sub_type","Ia_norm")

    catalog = _get_catalog()
    tinfo   = catalog.get(sub_type) if catalog else None

    # Load BOTH bands unconditionally for display, regardless of which band(s)
    # were actually used for fitting (active_bands may be 'primary' only) —
    # the chart should still show all available photometry for context.
    df_raw_all = load_all_bands_lc(obj)
    if df_raw_all is None: return jsonify({"error":"Could not load LC data."}), 400
    df_mod = apply_modifications(df_raw_all, obj)

    # Upper limits are intentionally NOT excluded here — the fit itself
    # (get_valid_fit_data) uses every point with a finite flux/flux_err
    # regardless of detection significance, since fitting happens in flux
    # space. The plot should show exactly what was fit.
    valid = df_mod[
        df_mod["flux"].notna() & np.isfinite(df_mod["flux"].values) & (df_mod["flux"] != 99.990) &
        df_mod["flux_err"].notna() & np.isfinite(df_mod["flux_err"].values) & (df_mod["flux_err"] > 0)
    ].copy()

    # "focus" is a narrow window around the fit — used only to compute the
    # *default* axis range (when the client has no prior zoom to preserve)
    # and to bound the smooth model curve. It never restricts which points
    # are actually sent: `window` below is the full dataset, unrestricted,
    # so panning/zooming after a fit can still reach every point that was
    # visible before the fit ran.
    if jd_min_fit is not None or jd_max_fit is not None:
        pad = 30.0
        t_min = (jd_min_fit - pad) if jd_min_fit is not None else best_time - 100.0
        t_max = (jd_max_fit + pad) if jd_max_fit is not None else best_time + 500.0
    else:
        t_min = best_time - 100.0; t_max = best_time + 500.0
    focus  = valid[(valid["JD"] >= t_min) & (valid["JD"] <= t_max)]
    window = valid.copy()

    # ── Data-only axis ranges (default camera position only) ───────────────
    data_x_range = None; data_y_range = None
    if not focus.empty:
        jd_vals = focus["JD"].values
        xpad = max((jd_vals.max() - jd_vals.min()) * 0.04, 2.0)
        if x_mode == 'ut':
            data_x_range = [_jd_to_ut(float(jd_vals.min() - xpad)),
                            _jd_to_ut(float(jd_vals.max() + xpad))]
        else:
            data_x_range = [round(float(jd_vals.min() - xpad), 3),
                            round(float(jd_vals.max() + xpad), 3)]
        fx_vals = focus["flux"].values
        fx_finite = fx_vals[np.isfinite(fx_vals)]
        if len(fx_finite):
            if show_flux:
                ypad = max(float(np.nanmax(fx_finite)) * 0.12, 1e-5)
                data_y_range = [round(float(min(0.0, float(np.nanmin(fx_finite)) - ypad*0.3)), 6),
                                round(float(np.nanmax(fx_finite) + ypad), 6)]
            else:
                fx_pos = fx_finite[fx_finite > 0]
                if len(fx_pos):
                    mag_bright = flux_to_mag_scalar(float(np.nanmax(fx_pos)))
                    mag_faint  = flux_to_mag_scalar(float(np.nanmin(fx_pos)))
                    pad = max((mag_faint - mag_bright) * 0.12, 0.1)
                    # Plotly range [faint+pad, bright-pad] renders bright at top
                    data_y_range = [round(mag_faint + pad, 3), round(mag_bright - pad, 3)]

    # A point is "in the fit" if it's in the JD range AND belongs to a band
    # that was actually used for fitting (active_bands may exclude one band).
    band_col_valid = window["band"].values if "band" in window.columns else np.array([fit_band] * len(window))
    in_range = np.isin(band_col_valid, list(fit_bands))
    if jd_min_fit is not None: in_range &= (window["JD"].values >= jd_min_fit)
    if jd_max_fit is not None: in_range &= (window["JD"].values <= jd_max_fit)

    traces = []

    # customdata: [flux_err, mag, mag_err, JD]  — JD always present for interaction handlers
    def _cd(mask):
        fl  = window["flux"].values[mask]; fe = window["flux_err"].values[mask]
        jds = window["JD"].values[mask]
        mag_arr = np.array([flux_to_mag_scalar(f) for f in fl])
        with np.errstate(divide='ignore',invalid='ignore'):
            me = np.where(fl>0, FLUX_TO_MAG_CONST*fe/fl, np.nan)
        return list(zip(fe.tolist(),
                        [None if not np.isfinite(v) else round(float(v),3) for v in mag_arr],
                        [None if not np.isfinite(v) else round(float(v),3) for v in me],
                        jds.tolist()))

    xl_lbl  = "UT" if x_mode == 'ut' else "JD"
    x_fmt   = "%{x}" if x_mode == 'ut' else "%{x:.2f}"
    y_fmt   = "%{y:.4f} mJy" if show_flux else "%{y:.3f} mag"
    y_lbl   = "Flux" if show_flux else "Mag"
    def _hov(b, cam, extra):
        return (f"<b>{b}-band {cam}{extra}</b><br>"
                f"{xl_lbl}: {x_fmt}<br>{y_lbl}: {y_fmt}<br>"
                "Flux: %{customdata[0]:.4f} mJy  Mag: %{customdata[1]:.3f}<extra></extra>")

    # Per-camera symbol map (same scheme as the viewer)
    CAM_SYMS = ['circle','square','diamond','cross','x','triangle-up','star','hexagram']
    cam_col  = window["camera"].values if "camera" in window.columns \
               else np.array(["unknown"] * len(window))
    unique_cams = sorted(set(cam_col))
    cam_sym  = {c: CAM_SYMS[i % len(CAM_SYMS)] for i, c in enumerate(unique_cams)}
    shown_cams: set = set()

    is_upper_col = window["upper_limit"].values if "upper_limit" in window.columns \
                   else np.zeros(len(window), dtype=bool)
    valid_mag_col = (window["mag"].notna() & (window["mag"] > 0) & (window["mag"] < 50)).values \
                    if "mag" in window.columns else np.ones(len(window), dtype=bool)

    # ── Detections + upper limits, per band + camera — nothing is ever
    # hidden. Full opacity if actually used in the fit (right band, right JD
    # range), dimmed otherwise. In flux mode upper limits are indistinguishable
    # from detections (both have a real flux measurement — same marker, same
    # code path as the initial light-curve load). In magnitude mode they keep
    # the standard triangle-down convention and use the recorded limiting
    # magnitude directly (deriving it from flux would drop points whose
    # background-subtracted flux is <= 0).
    for b in ("g", "V"):
        b_mask_all = (band_col_valid == b)
        if not b_mask_all.any(): continue
        bcolor, bdim = BAND_COLOR.get(b, "#e2e8f0"), BAND_DIM.get(b, "rgba(226,232,240,0.3)")
        for cam in unique_cams:
            cam_mask_base = b_mask_all & (cam_col == cam)
            if show_flux:
                groups = [(cam_mask_base, False)]
            else:
                groups = [(cam_mask_base & ~is_upper_col, False),
                          (cam_mask_base & is_upper_col & valid_mag_col, True)]
            for base, is_ul in groups:
                if not base.any(): continue
                sym    = "triangle-down" if is_ul else cam_sym[cam]
                ul_lbl = " lim." if is_ul else ""
                for bright in (True, False):
                    mask = base & (in_range if bright else ~in_range)
                    if not mask.any(): continue
                    sub   = window[mask]
                    color = bcolor if bright else bdim
                    trace = {
                        "type":"scatter","x":_xl(sub["JD"].tolist()),
                        "mode":"markers","name":cam,"legendgroup":cam,
                        "showlegend": cam not in shown_cams,
                        "marker":{"color":color,"symbol":sym,
                                  "size":12 if bright else 10,"opacity":0.9 if bright else 0.35},
                        "customdata":_cd(mask),"xaxis":"x","yaxis":"y",
                    }
                    if is_ul:
                        trace["y"] = sub["mag"].tolist()
                        trace["hovertemplate"] = (
                            f"<b>{b}-band {cam}{ul_lbl}{'' if bright else ' (not in fit)'}</b><br>"
                            f"{xl_lbl}: {x_fmt}<br>Mag lim: %{{y:.3f}}<extra></extra>")
                    else:
                        fl = sub["flux"].values; fe = sub["flux_err"].values
                        trace["y"] = [_y(v) for v in fl]
                        trace["error_y"] = {"type":"data","array":_yerr(fl,fe),"visible":True,
                                             "thickness":1.5 if bright else 1,
                                             "width":5 if bright else 4,"color":color}
                        trace["hovertemplate"] = _hov(b, cam, "" if bright else " (not in fit)")
                    traces.append(trace)
                    shown_cams.add(cam)

    # ── Model curve ───────────────────────────────────────────────────────
    smooth_t = np.linspace(t_min, t_max, 600)
    if tinfo is not None:
        p_model = [best_flux, best_time]
        if tinfo['type'] == '2D' and res.get("sBV") is not None:
            p_model.append(float(res["sBV"]))
        elif tinfo['type'] == 'CV' and res.get("stretch") is not None:
            p_model.append(float(res["stretch"]))
        model_flux_arr = model_fluxes(p_model, smooth_t, redshift, tinfo, fit_band)
    else:
        model_flux_arr = np.zeros_like(smooth_t)
    model_y_list = _yl([None if not np.isfinite(v) else float(v) for v in model_flux_arr])
    mod_hov = (f"{xl_lbl}: {x_fmt}<br>Model: "
               + ("%{y:.4f} mJy" if show_flux else "%{y:.3f} mag") + "<extra></extra>")
    traces.append({
        "type":"scatter","x":_xl(smooth_t.tolist()),"y":model_y_list,
        "mode":"lines","name":f"Best-fit ({sub_type})",
        "line":{"color":"#ef4444","width":2.5},
        "hovertemplate":mod_hov,
        "xaxis":"x","yaxis":"y",
    })

    # ── MC samples ────────────────────────────────────────────────────────
    mc_pf  = res.get("mc_peak_flux") or []
    mc_pt  = res.get("mc_peak_time") or []
    mc_sBV = res.get("mc_sBV") or []
    smooth_x = _xl(smooth_t.tolist())
    n_mc_shown = 0
    if mc_pf and tinfo is not None:
        n_plot = min(60, len(mc_pf))
        idxs   = np.random.choice(len(mc_pf), size=n_plot, replace=False)
        for ii, si in enumerate(idxs):
            try:
                pm = [float(mc_pf[si]), float(mc_pt[si])]
                if tinfo['type'] == '2D' and mc_sBV and si < len(mc_sBV):
                    pm.append(float(mc_sBV[si]))
                elif tinfo['type'] == 'CV':
                    pm.append(float(res.get("stretch", 1.0) or 1.0))
                mc_flux = model_fluxes(pm, smooth_t, redshift, tinfo, fit_band)
                y_list = _yl([None if not np.isfinite(v) else float(v) for v in mc_flux])
                traces.append({
                    "type":"scatter","x":smooth_x,"y":y_list,
                    "mode":"lines","name":"MC sample" if n_mc_shown==0 else "",
                    "line":{"color":"rgba(150,150,150,0.35)","width":1.2},
                    "showlegend":n_mc_shown==0,"hoverinfo":"skip",
                    "xaxis":"x","yaxis":"y",
                })
                n_mc_shown += 1
            except Exception:
                continue

    # ── Residuals (per camera, fit band + fit range only) ─────────────────
    if not window.empty and in_range.any() and tinfo is not None:
        fit_sub = window[in_range]; t_sub = fit_sub["JD"].values
        p_m = [best_flux, best_time]
        if tinfo['type'] == '2D' and res.get("sBV"): p_m.append(float(res["sBV"]))
        elif tinfo['type'] == 'CV' and res.get("stretch"): p_m.append(float(res["stretch"]))
        model_at  = model_fluxes(p_m, t_sub, redshift, tinfo, fit_band)
        residuals = (fit_sub["flux"].values - model_at) / fit_sub["flux_err"].values
        resid_hov = f"{xl_lbl}: {x_fmt}<br>Residual: %{{y:.2f}} σ<extra></extra>"
        fit_cam_col = fit_sub["camera"].values if "camera" in fit_sub.columns \
                      else np.array(["unknown"] * len(fit_sub))
        resid_color = BAND_COLOR.get(fit_band, "#10b981")
        for cam in unique_cams:
            cm = fit_cam_col == cam
            if not cm.any(): continue
            traces.append({
                "type":"scatter","x":_xl(fit_sub["JD"].values[cm].tolist()),
                "y":residuals[cm].tolist(),
                "mode":"markers","name":cam,"legendgroup":cam,"showlegend":False,
                "marker":{"color":resid_color,"symbol":cam_sym.get(cam,"circle"),"size":8,"opacity":0.75},
                "hovertemplate":resid_hov,
                "xaxis":"x","yaxis":"y2",
            })
        traces.append({"type":"scatter","x":_xl([t_min,t_max]),"y":[0,0],"mode":"lines",
                        "line":{"color":"#6b7280","width":1,"dash":"dash"},
                        "showlegend":False,"hoverinfo":"skip","xaxis":"x","yaxis":"y2"})

    shapes = [{"type":"line","x0":_x(best_time),"x1":_x(best_time),"y0":0,"y1":1,"yref":"paper",
               "line":{"color":"#ef4444","width":1.5,"dash":"dot"}}]
    if disc_jd:
        shapes.append({"type":"line","x0":_x(disc_jd),"x1":_x(disc_jd),"y0":0,"y1":1,"yref":"paper",
                        "line":{"color":"#22d3ee","width":1.5,"dash":"dash"}})
    for jv in (jd_min_fit, jd_max_fit):
        if jv is not None:
            shapes.append({"type":"line","x0":_x(jv),"x1":_x(jv),"y0":0,"y1":1,"yref":"paper",
                            "line":{"color":"#a78bfa","width":1,"dash":"dot"}})

    return jsonify({
        "traces":traces,"shapes":shapes,"name":name,
        "peak_flux":best_flux,"app_mag":res.get("app_mag"),"abs_mag":res.get("abs_mag"),
        "peak_time":best_time,"disc_jd":disc_jd,"chi2_red":res.get("chi2_red"),
        "n_pts":res.get("n_pts"),"sub_type":sub_type,
        "jd_min_fit":jd_min_fit,"jd_max_fit":jd_max_fit,
        "has_mc": bool(res.get("mc_peak_flux")),
        "n_mc_shown": n_mc_shown,
        "data_x_range": data_x_range,
        "data_y_range": data_y_range,
        "xaxis_title": xaxis_title,
        "yaxis_title": yaxis_title,
        "x_mode": x_mode,
        "show_flux": show_flux,
    })


# ── Fitter data-manipulation routes ───────────────────────────────────────────

def _get_fit_obj(data=None, idx=None):
    n = len(OBJECTS)
    if n == 0: return None, -1
    if idx is None:
        d = data or request.json or {}
        idx = max(0, min(int(d.get("index", FIT_STATE["current"])), n-1))
    FIT_STATE["current"] = idx
    return OBJECTS[idx], idx


@app.route("/api/fit/remove_point", methods=["POST"])
def api_fit_remove_point():
    data = request.json or {}; obj, idx = _get_fit_obj(data)
    if obj is None: return jsonify({"error":"no objects"}), 400
    jd = float(data.get("jd"))
    _push_undo(obj)
    obj["removed_jds"].append(jd)
    return jsonify({"status":"ok","removed_jds":obj["removed_jds"]})


@app.route("/api/fit/remove_region", methods=["POST"])
def api_fit_remove_region():
    data = request.json or {}; obj, idx = _get_fit_obj(data)
    if obj is None: return jsonify({"error":"no objects"}), 400
    x1=float(data["x1"]); x2=float(data["x2"])
    y1=float(data["y1"]); y2=float(data["y2"])
    jd_lo=min(x1,x2); jd_hi=max(x1,x2)
    fl_lo=min(y1,y2); fl_hi=max(y1,y2)
    _, df_mod, _ = get_valid_fit_data(obj)
    if df_mod is None: return jsonify({"status":"ok","n_removed":0}), 200
    region = df_mod[
        (df_mod["JD"] >= jd_lo) & (df_mod["JD"] <= jd_hi) &
        (df_mod["flux"] >= fl_lo) & (df_mod["flux"] <= fl_hi)
    ]
    if region.empty: return jsonify({"status":"ok","n_removed":0}), 200
    _push_undo(obj)
    for jd in region["JD"].values:
        if not any(np.isclose(jd, r, rtol=0, atol=1e-5) for r in obj["removed_jds"]):
            obj["removed_jds"].append(float(jd))
    return jsonify({"status":"ok","n_removed":len(region),"removed_jds":obj["removed_jds"]})


@app.route("/api/fit/reset_points", methods=["POST"])
def api_fit_reset_points():
    data = request.json or {}; obj, idx = _get_fit_obj(data)
    if obj is None: return jsonify({"error":"no objects"}), 400
    _push_undo(obj)
    obj["removed_jds"] = []
    return jsonify({"status":"ok"})


@app.route("/api/fit/set_v_scale", methods=["POST"])
def api_fit_set_v_scale():
    data = request.json or {}; obj, idx = _get_fit_obj(data)
    if obj is None: return jsonify({"error":"no objects"}), 400
    try: scale = float(data.get("v_scale",1.0))
    except: return jsonify({"error":"v_scale must be a number"}), 400
    _push_undo(obj); obj["v_scale"] = scale
    return jsonify({"status":"ok","v_scale":scale})


@app.route("/api/fit/set_error_inflate", methods=["POST"])
def api_fit_set_error_inflate():
    data = request.json or {}; obj, idx = _get_fit_obj(data)
    if obj is None: return jsonify({"error":"no objects"}), 400
    try: val = float(data.get("error_inflate",0.0))
    except: return jsonify({"error":"error_inflate must be a number"}), 400
    if val < 0: return jsonify({"error":"error_inflate cannot be negative"}), 400
    _push_undo(obj); obj["error_inflate"] = val
    return jsonify({"status":"ok","error_inflate":val})


@app.route("/api/fit/apply_baseline_shift", methods=["POST"])
def api_fit_apply_baseline_shift():
    data = request.json or {}; obj, idx = _get_fit_obj(data)
    if obj is None: return jsonify({"error":"no objects"}), 400
    try:
        jd_start = float(data.get("jd_start")); jd_end = float(data.get("jd_end"))
    except: return jsonify({"error":"jd_start and jd_end must be numbers"}), 400

    df_raw = load_combined_lc(obj)
    if df_raw is None: return jsonify({"error":"No LC data available"}), 400

    df_mod = apply_modifications(df_raw, obj)
    window = df_mod[(df_mod["JD"] >= jd_start) & (df_mod["JD"] <= jd_end) &
                    df_mod["flux"].notna() & np.isfinite(df_mod["flux"].values)]
    if window.empty: return jsonify({"error":"No data in specified JD range"}), 400

    camera_shifts = window.groupby("camera")["flux"].median().to_dict()
    _push_undo(obj)
    for cam, shift in camera_shifts.items():
        obj["baseline_shifts"][cam] = obj["baseline_shifts"].get(cam, 0.0) + float(shift)
    return jsonify({"status":"ok","camera_shifts":camera_shifts,
                    "all_baseline_shifts":obj["baseline_shifts"]})


@app.route("/api/fit/filter_by_error", methods=["POST"])
def api_fit_filter_by_error():
    data = request.json or {}; obj, idx = _get_fit_obj(data)
    if obj is None: return jsonify({"error":"no objects"}), 400
    try: n_sigma = float(data.get("n_sigma", data.get("n_factor", 3.0)))
    except: return jsonify({"error":"n_sigma must be a number"}), 400

    _, df_mod, valid = get_valid_fit_data(obj)
    if valid is None or valid.empty: return jsonify({"error":"No valid data"}), 400

    median_err = valid["flux_err"].median()
    rm_mask    = valid["flux_err"] > n_sigma * median_err
    to_remove  = valid[rm_mask]["JD"].tolist()
    if not to_remove: return jsonify({"status":"ok","n_removed":0}), 200

    _push_undo(obj)
    for jd in to_remove:
        if not any(np.isclose(jd, r, rtol=0, atol=1e-5) for r in obj["removed_jds"]):
            obj["removed_jds"].append(float(jd))
    return jsonify({"status":"ok","n_removed":len(to_remove),"removed_jds":obj["removed_jds"]})


@app.route("/api/fit/set_redshift", methods=["POST"])
def api_fit_set_redshift():
    data = request.json or {}; obj, idx = _get_fit_obj(data)
    if obj is None: return jsonify({"error":"no objects"}), 400
    try: z_str = data.get("redshift",""); z = float(z_str) if z_str.strip() else float('nan')
    except: return jsonify({"error":"Redshift must be a number"}), 400
    _push_undo(obj)
    obj["redshift"] = z if np.isfinite(z) else None
    obj["dist_mod"] = compute_dist_mod(z) if np.isfinite(z) else None
    return jsonify({"status":"ok","redshift":_safe(z),"dist_mod":_safe(obj["dist_mod"])})


@app.route("/api/fit/set_bands", methods=["POST"])
def api_fit_set_bands():
    data = request.json or {}; obj, idx = _get_fit_obj(data)
    if obj is None: return jsonify({"error":"no objects"}), 400
    active = data.get("active_bands","primary")
    band   = data.get("band")
    if active not in ("primary","both"): return jsonify({"error":"active_bands must be 'primary' or 'both'"}), 400
    if band is not None and band not in ("g","V"): return jsonify({"error":"band must be 'g' or 'V'"}), 400
    _push_undo(obj)
    obj["active_bands"] = active
    if band is not None: obj["band"] = band
    return jsonify({"status":"ok","active_bands":active,"band":obj["band"]})


@app.route("/api/fit/set_flag", methods=["POST"])
def api_fit_set_flag():
    data = request.json or {}; obj, idx = _get_fit_obj(data)
    if obj is None: return jsonify({"error":"no objects"}), 400
    flag = data.get("flag")
    _push_undo(obj); obj["flag"] = flag
    return jsonify({"status":"ok","flag":flag,"name":obj["name"]})


@app.route("/api/fit/undo", methods=["POST"])
def api_fit_undo():
    data = request.json or {}; obj, idx = _get_fit_obj(data)
    if obj is None: return jsonify({"error":"no objects"}), 400
    ok = _pop_undo(obj)
    if not ok: return jsonify({"error":"Nothing to undo"}), 400
    return jsonify({"status":"ok","has_undo":len(obj["undo_stack"])>0})


@app.route("/api/fit/revert", methods=["POST"])
def api_fit_revert():
    data = request.json or {}; obj, idx = _get_fit_obj(data)
    if obj is None: return jsonify({"error":"no objects"}), 400
    _push_undo(obj)
    _revert(obj)
    return jsonify({"status":"ok"})


@app.route("/api/fit/toggle_complete", methods=["POST"])
def api_fit_toggle_complete():
    data = request.json or {}; obj, idx = _get_fit_obj(data)
    if obj is None: return jsonify({"error":"no objects"}), 400
    name = obj["name"]
    if name in FIT_STATE["completed"]: FIT_STATE["completed"].discard(name); is_complete=False
    else:                               FIT_STATE["completed"].add(name);     is_complete=True
    return jsonify({"name":name,"completed":is_complete,"index":idx})


@app.route("/api/fit/navigate", methods=["POST"])
def api_fit_navigate():
    data = request.json or {}; n = len(OBJECTS)
    if n == 0: return jsonify({"error":"no objects"}), 400
    target = data.get("target")
    idx = max(0, min(int(target), n-1)) if target is not None else \
          (FIT_STATE["current"] + data.get("direction",0)) % n
    FIT_STATE["current"] = idx
    return jsonify({"index":idx,"total":n})


@app.route("/api/fit/save_result", methods=["POST"])
def api_fit_save_result():
    data        = request.json or {}
    obj, idx    = _get_fit_obj(data)
    if obj is None: return jsonify({"error":"no objects"}), 400
    output_path = data.get("output_path","").strip() or FIT_STATE.get("output_csv","").strip()
    if not output_path: return jsonify({"error":"No output CSV path specified"}), 400
    name   = obj["name"]
    result = FIT_STATE["results"].get(name)
    if result is None: return jsonify({"error":f"No fit result for {name} — run fit first."}), 400

    # Build output row
    meta = obj.get("meta",{})
    row  = {}
    # Identifying info
    row["Name"] = name
    for c in ("TNS_Name_clean","SNName","RA_deg","DEC_deg","ra_hms","dec_dms",
               "redshift_to_use","type_to_use","Discovery Date (UT)","sample"):
        if c in meta: row[c] = meta[c]
    if "redshift_to_use" not in row:
        row["redshift_to_use"] = _safe(obj.get("redshift"))
    row["dist_mod"]      = _safe(obj.get("dist_mod"))
    row["k_corr"]        = _safe(obj.get("k_corr",0.0))
    row["gal_ext"]       = _safe(obj.get("gal_ext",0.0))
    # Fit results
    for k in ("sub_type","peak_flux","peak_flux_err_low","peak_flux_err_up",
              "peak_time","peak_time_err_low","peak_time_err_up",
              "sBV","sBV_err_low","sBV_err_up",
              "stretch","app_mag","app_mag_err_low","app_mag_err_up",
              "abs_mag","abs_mag_err_low","abs_mag_err_up",
              "abs_mag_sBV","chi2_red","n_pts",
              "jd_min_fit","jd_max_fit","disc_jd"):
        row[k] = result.get(k)
    row["flag"]        = obj.get("flag")
    row["Completed"]   = name in FIT_STATE["completed"]
    row["Removed_JDs"] = json.dumps(obj.get("removed_jds",[]))
    row["v_scale"]     = obj.get("v_scale",1.0)
    row["error_inflate"] = obj.get("error_inflate",0.0)
    row["warnings"]    = "|".join(result.get("warnings",[]))

    # Replace None with np.nan so pandas writes NaN cleanly for all numeric cols
    clean_row = {k: (np.nan if v is None else v) for k, v in row.items()}

    out = Path(output_path)
    try:
        new_df = pd.DataFrame([clean_row])
        if out.exists():
            ex_df = pd.read_csv(out)
            if "Name" in ex_df.columns and name in ex_df["Name"].values:
                # Drop old row, append updated one — avoids dtype conflicts from loc assignment
                ex_df = ex_df[ex_df["Name"] != name].reset_index(drop=True)
                ex_df = pd.concat([ex_df, new_df], ignore_index=True)
                ex_df.to_csv(out, index=False)
                return jsonify({"status":"updated","name":name,"path":str(out)})
            else:
                pd.concat([ex_df, new_df], ignore_index=True).to_csv(out, index=False)
                return jsonify({"status":"appended","name":name,"path":str(out)})
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            new_df.to_csv(out, index=False)
            return jsonify({"status":"created","name":name,"path":str(out)})
    except Exception as e:
        return jsonify({"error":f"Save error: {e}"}), 500


# ── Entry point ───────────────────────────────────────────────────────────────

def _open_browser(port):
    time.sleep(0.9)
    webbrowser.open(f"http://127.0.0.1:{port}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ASAS-SN Light Curve Viewer + Fitter")
    parser.add_argument("--port",       type=int, default=5050)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    print(f"\n  ASAS-SN LC Viewer + Fitter v2\n  Running at: http://127.0.0.1:{args.port}\n  Ctrl+C to stop.\n")
    if not args.no_browser:
        threading.Thread(target=_open_browser, args=(args.port,), daemon=True).start()
    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)
