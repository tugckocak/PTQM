"""PESQM — Perceptual Echo and Sidetone Quality Measure.

Implementation of the objective talking-quality model from:
  Appel & Beerends, "Objective Measurement of the Quality with Which You
  Perceive Your Own Voice" (talking quality), JAES vol. 50 no. 4, 2002,
  Section 2.3 / Fig. 5.

Pipeline (no time alignment, by design — echo must remain delayed):
  ref x[t], deg y[t]
    -> 32 ms Hanning frames, 50% overlap, FFT power spectrum
    -> frequency warping to pitch (Bark) scale, ~0.31 Bark bands
    -> level scaling to 79 dB SPL equivalent listening level
    -> excitation: upward spread of masking,
         s = 22 + 230/f_m - 0.2*P[dB]  (dB / critical band, max 32)
         E[j] = ( sum_{mu<=j} PE[mu->j]^0.8 )^1.25
    -> Zwicker loudness density
         L[j] = Sl*(P0/0.5)^0.23 * ( (0.5 + 0.5*E/P0)^0.23 - 1 )
    -> perceptual subtraction  N[j] = |Ly[j] - Lx[j]| - 0.01  (>=0)
    -> noise-masking threshold: frame disturbance kept only when
         Ly_i > 1.6 * max( min_i Ly_i , 0.5 sone )
    -> NSR_i = ( mean_j (Nthr[j]/Ly_i)^1.4 )^(1/1.4)     (Lp = 1.4)
    -> PESQM = ( mean_i NSR_i^5 )^(1/5)                  (Lq = 5)
"""

import numpy as np

SR = 16000
FRAME = 512            # 32 ms @ 16 kHz
HOP = 256              # 50% overlap
DZ = 0.31              # Bark band width
LISTENING_LEVEL_DB = 79.0   # active speech scaled to 79 dB SPL equivalent
INTERNAL_NOISE = 0.01       # sone, subtracted from loudness difference
SONE_FLOOR = 0.5            # sone, minimum-loudness floor
THRESH_FACTOR = 1.6
LP = 1.4                    # frequency norm
LQ = 5.0                    # time norm


def _bark(f):
    f = np.asarray(f, dtype=np.float64)
    return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


def _threshold_quiet_db(f):
    """Absolute threshold in quiet (dB SPL), Terhardt approximation."""
    fk = np.maximum(np.asarray(f, dtype=np.float64), 20.0) / 1000.0
    return (3.64 * fk ** -0.8
            - 6.5 * np.exp(-0.6 * (fk - 3.3) ** 2)
            + 1e-3 * fk ** 4)


class _Bands:
    """Precomputed Bark-band layout for the FFT grid."""

    def __init__(self, sr=SR, nfft=FRAME):
        freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
        zmax = _bark(sr / 2.0)
        edges = np.arange(DZ, zmax + DZ, DZ)          # band upper edges
        self.n_bands = len(edges)
        zf = _bark(freqs)
        # map every FFT bin (>0 Hz) to a band index
        self.bin_band = np.clip(np.searchsorted(edges, zf), 0, self.n_bands - 1)
        self.bin_band[0] = -1                          # drop DC
        # band centres (Bark and Hz)
        self.z_centre = edges - DZ / 2.0
        # invert bark -> Hz numerically for centres
        fgrid = np.linspace(1.0, sr / 2.0, 4000)
        self.f_centre = np.interp(self.z_centre, _bark(fgrid), fgrid)
        self.p0_lin = 10.0 ** (_threshold_quiet_db(self.f_centre) / 10.0)

    def pool(self, power_bins):
        """Sum FFT-bin powers into Bark bands. power_bins: (frames, bins)."""
        out = np.zeros((power_bins.shape[0], self.n_bands))
        valid = self.bin_band >= 0
        np.add.at(out.T, self.bin_band[valid], power_bins[:, valid].T)
        return out


_BANDS = _Bands()
_SL = None   # loudness scaling, calibrated on first use


def _frames_power(x):
    """Hanning-windowed FFT power per frame. Returns (n_frames, n_bins)."""
    n = (len(x) - FRAME) // HOP + 1
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    win = np.hanning(FRAME)
    spec = np.fft.rfft(x[idx] * win, axis=1)
    # normalise so that a full-scale sine has power ~0.5 regardless of window
    return (np.abs(spec) ** 2) * (2.0 / (win.sum() ** 2))


def _active_power(x):
    """Mean power over active frames (within 35 dB of the 95th percentile)."""
    n = max((len(x) - FRAME) // HOP + 1, 1)
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    p = (x[idx] ** 2).mean(axis=1)
    ref = np.percentile(p[p > 0], 95)
    act = p[p > ref * 10 ** (-35 / 10)]
    return act.mean() if len(act) else p.mean()


def _excitation(P_lin):
    """Upward spread of masking. P_lin: (frames, bands) linear power.

    Level-dependent slope s = 22 + 230/f_m - 0.2*P_dB (dB/Bark), <= 32.
    E[j] = ( sum_mu PE[mu->j]^0.8 )^1.25, PE[j,j] = own power.
    """
    nfr, nb = P_lin.shape
    P_db = 10.0 * np.log10(np.maximum(P_lin, 1e-12))
    s = 22.0 + 230.0 / _BANDS.f_centre[None, :] - 0.2 * P_db   # (frames, bands)
    s = np.clip(s, 2.0, 32.0)
    dz = DZ * (np.arange(nb)[None, :] - np.arange(nb)[:, None])  # (mu, j) Bark
    dz[dz < 0] = np.inf                                          # upward only
    # att[frame, mu, j] = 10^(-s[frame,mu]*dz[mu,j]/10)
    att = 10.0 ** (-(s[:, :, None] * dz[None, :, :]) / 10.0)
    pe = (P_lin[:, :, None] * att) ** 0.8
    return pe.sum(axis=1) ** 1.25


def _loudness(P_lin):
    """Zwicker loudness density (sone/Bark) per frame/band."""
    global _SL
    if _SL is None:
        _SL = 1.0
        _SL = 1.0 / _calibration_total_loudness()
    E = _excitation(P_lin)
    p0 = _BANDS.p0_lin[None, :]
    L = _SL * (p0 / 0.5) ** 0.23 * ((0.5 + 0.5 * E / p0) ** 0.23 - 1.0)
    return np.maximum(L, 0.0)


def _calibration_total_loudness():
    """Total loudness of a 1 kHz sinusoid at 40 dB SPL (must map to 1 sone)."""
    t = np.arange(SR) / SR
    x = np.sqrt(2.0) * np.sin(2 * np.pi * 1000.0 * t)
    # scale so total band power equals a 40 dB SPL intensity
    P = _BANDS.pool(_frames_power(x))
    P *= (10.0 ** (40.0 / 10.0)) / P.sum(axis=1).mean()
    L = _loudness_raw(P, sl=1.0)
    return (L.sum(axis=1) * DZ).mean()


def _loudness_raw(P_lin, sl):
    E = _excitation(P_lin)
    p0 = _BANDS.p0_lin[None, :]
    L = sl * (p0 / 0.5) ** 0.23 * ((0.5 + 0.5 * E / p0) ** 0.23 - 1.0)
    return np.maximum(L, 0.0)


def _disturbance(ref, deg, level_db, noise_floor_pct):
    """Shared pipeline front half: gated disturbance density N (frames x
    bands, sone/Bark) and degraded frame loudness Ly_i (sone)."""
    ref = np.asarray(ref, dtype=np.float64)
    deg = np.asarray(deg, dtype=np.float64)
    n = min(len(ref), len(deg))
    ref, deg = ref[:n], deg[:n]

    # common level calibration from the reference
    gain = 10.0 ** (level_db / 20.0) / np.sqrt(_active_power(ref))
    ref = ref * gain
    deg = deg * gain

    Px = _BANDS.pool(_frames_power(ref))
    Py = _BANDS.pool(_frames_power(deg))

    if noise_floor_pct is not None:
        Px = Px + np.percentile(Py, noise_floor_pct, axis=0)[None, :]

    Lx = _loudness(Px)
    Ly = _loudness(Py)

    # perceptual subtraction -> disturbance density
    N = np.maximum(np.abs(Ly - Lx) - INTERNAL_NOISE, 0.0)

    # frame loudness of the degraded signal (frequency integration)
    Ly_i = Ly.sum(axis=1) * DZ
    thresh = THRESH_FACTOR * max(Ly_i.min(), SONE_FLOOR)
    N[Ly_i <= thresh, :] = 0.0
    return N, Ly_i


def pesqm(ref, deg, sr=SR, level_db=LISTENING_LEVEL_DB, noise_floor_pct=10):
    """PESQM score. Higher = more audible disturbance = worse talking quality.

    ref: clean own-voice / sidetone signal
    deg: same signal plus returned echo / distortion (NOT time-aligned)
    Both scaled together: ref active speech level -> level_db dB SPL
    equivalent (default 79).

    noise_floor_pct: deviation from the paper. The stationary noise floor of
    the degraded signal (per-band percentile over frames) is added to the
    reference before the loudness stage, so steady background noise is part
    of the listener's implicit reference and partially masks the echo
    (Sec 1.5 of the paper: louder noise -> better talking quality). Validated
    range 5-20; set to None to disable and get the literal paper pipeline.
    """
    assert sr == SR, "resample to 16 kHz first"
    N, Ly_i = _disturbance(ref, deg, level_db, noise_floor_pct)

    # NSR per frame: local disturbance-to-signal-loudness ratio, Lp over bands
    denom = np.maximum(Ly_i, SONE_FLOOR)[:, None]
    nsr = ((N / denom) ** LP).mean(axis=1) ** (1.0 / LP)

    # Lq norm over time
    return float((nsr ** LQ).mean() ** (1.0 / LQ))


def disturbance_map(ref, deg, sr=SR, level_db=LISTENING_LEVEL_DB,
                    noise_floor_pct=10, normalized=True):
    """Time x Bark disturbance representation N_i[j] for model input.

    Returns (frames, n_bands) float32: the gated loudness-difference density
    between deg and ref (32 ms frames, 16 ms hop, 0.31-Bark bands).
    normalized=True divides each frame by its degraded loudness (the local
    disturbance-to-signal ratio that PESQM's norms aggregate), so the map is
    level-robust; the caller applies its own compression (e.g. log1p).
    """
    assert sr == SR, "resample to 16 kHz first"
    N, Ly_i = _disturbance(ref, deg, level_db, noise_floor_pct)
    if normalized:
        N = N / np.maximum(Ly_i, SONE_FLOOR)[:, None]
    return N.astype(np.float32)
