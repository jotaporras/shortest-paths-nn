"""Irregular node samples of a DEM grid.

Every sampler takes the full-resolution DEM arrays ``xv, yv, elev`` (km, shape
(rows, cols), as returned by ``grid.get_dem_xv_yv_``) and returns

- ``node_features``: (N, 3) array of [x, y, z] in km, and
- ``dem_idx``: (N,) flat indices into the DEM (row * cols + col), sorted.

Each node is a real DEM cell, so no elevation is interpolated.
"""
import numpy as np


def _features(xv, yv, elev, dem_idx):
    dem_idx = np.sort(np.asarray(dem_idx, dtype=np.int64))
    feats = np.stack([xv.ravel()[dem_idx], yv.ravel()[dem_idx], elev.ravel()[dem_idx]], axis=1)
    return feats, dem_idx


def sample_regular_grid(xv, yv, elev, stride):
    """Every ``stride``-th row and column, the scheme of ``make_dataset --resolution``."""
    rows, cols = elev.shape
    r, c = np.meshgrid(np.arange(0, rows, stride), np.arange(0, cols, stride), indexing='ij')
    return _features(xv, yv, elev, (r * cols + c).ravel())


def sample_uniform(xv, yv, elev, n, seed=0):
    """``n`` DEM cells drawn uniformly at random without replacement."""
    rng = np.random.default_rng(seed)
    return _features(xv, yv, elev, rng.choice(elev.size, size=n, replace=False))


def slope_magnitude(xv, yv, elev):
    """|grad z| per DEM cell (dimensionless, rise over run)."""
    dx = xv[0, 1] - xv[0, 0]
    dy = yv[1, 0] - yv[0, 0]
    gz_y, gz_x = np.gradient(elev, dy, dx)
    return np.hypot(gz_x, gz_y)


def laplacian_magnitude(xv, yv, elev):
    """|Laplacian of z| per DEM cell (1/km)."""
    dx = xv[0, 1] - xv[0, 0]
    dy = yv[1, 0] - yv[0, 0]
    gz_y, gz_x = np.gradient(elev, dy, dx)
    return np.abs(np.gradient(gz_x, dx, axis=1) + np.gradient(gz_y, dy, axis=0))


def slope_adaptive_density(xv, yv, elev, uniform_mix=0.5, feature='slope'):
    """Sampling density over DEM cells, normalized to sum to 1.

    rho = uniform_mix / n_cells + (1 - uniform_mix) * f / sum(f), with f the
    slope or |Laplacian| magnitude. ``uniform_mix > 0`` keeps rho >= rho_min > 0.
    """
    if not 0.0 <= uniform_mix <= 1.0:
        raise ValueError(f"uniform_mix must be in [0, 1], got {uniform_mix}")
    if feature == 'slope':
        f = slope_magnitude(xv, yv, elev)
    elif feature == 'laplacian':
        f = laplacian_magnitude(xv, yv, elev)
    else:
        raise ValueError(f"unknown feature {feature!r}; use 'slope' or 'laplacian'")
    f = f.ravel()
    total = f.sum()
    adaptive = f / total if total > 0 else np.full(f.size, 1.0 / f.size)
    rho = uniform_mix / f.size + (1.0 - uniform_mix) * adaptive
    return rho / rho.sum()


def sample_slope_adaptive(xv, yv, elev, n, uniform_mix=0.5, feature='slope', seed=0):
    """``n`` DEM cells drawn without replacement with density ``slope_adaptive_density``."""
    rng = np.random.default_rng(seed)
    rho = slope_adaptive_density(xv, yv, elev, uniform_mix=uniform_mix, feature=feature)
    return _features(xv, yv, elev, rng.choice(elev.size, size=n, replace=False, p=rho))


def _bridson(width, height, min_dist, rng, k=30):
    """Bridson's Poisson-disk sampling on [0, width] x [0, height]."""
    cell = min_dist / np.sqrt(2.0)
    gw = int(np.ceil(width / cell)) + 1
    gh = int(np.ceil(height / cell)) + 1
    grid = -np.ones((gw, gh), dtype=np.int64)
    pts = []

    def add(p):
        grid[int(p[0] / cell), int(p[1] / cell)] = len(pts)
        pts.append(p)

    def far_enough(p):
        gx, gy = int(p[0] / cell), int(p[1] / cell)
        for i in range(max(gx - 2, 0), min(gx + 3, gw)):
            for j in range(max(gy - 2, 0), min(gy + 3, gh)):
                q = grid[i, j]
                if q >= 0 and (pts[q][0] - p[0]) ** 2 + (pts[q][1] - p[1]) ** 2 < min_dist ** 2:
                    return False
        return True

    add(np.array([rng.uniform(0, width), rng.uniform(0, height)]))
    active = [0]
    while active:
        a = rng.integers(len(active))
        base = pts[active[a]]
        radius = rng.uniform(min_dist, 2 * min_dist, size=k)
        angle = rng.uniform(0, 2 * np.pi, size=k)
        cand = base + np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)
        placed = False
        for p in cand:
            if 0 <= p[0] <= width and 0 <= p[1] <= height and far_enough(p):
                add(p)
                active.append(len(pts) - 1)
                placed = True
                break
        if not placed:
            active[a] = active[-1]
            active.pop()
    return np.array(pts)


def sample_poisson_disk(xv, yv, elev, min_dist=None, n_target=None, seed=0, k=30, calib_iters=2):
    """Poisson-disk (blue-noise) sample: no two nodes closer than ``min_dist`` km in xy.

    Bridson's algorithm runs in the continuous domain; each point then snaps to
    its nearest DEM cell (duplicates dropped). Snapping moves a point by at most
    half a cell, so the realized minimum distance is min_dist - one cell diagonal at worst.

    Pass ``min_dist`` directly, or ``n_target`` to calibrate it: the run starts at
    d = sqrt(area / n_target) (the spacing of a regular grid with n_target nodes) and
    rescales d by sqrt(N / n_target) for ``calib_iters`` more runs.
    Returns (node_features, dem_idx, min_dist_used).
    """
    if (min_dist is None) == (n_target is None):
        raise ValueError("pass exactly one of min_dist and n_target")
    rng = np.random.default_rng(seed)
    rows, cols = elev.shape
    x0, y0 = xv[0, 0], yv[0, 0]
    width, height = xv[0, -1] - x0, yv[-1, 0] - y0
    dx, dy = xv[0, 1] - x0, yv[1, 0] - y0

    def run(d):
        pts = _bridson(width, height, d, rng, k=k)
        c = np.clip(np.rint(pts[:, 0] / dx).astype(np.int64), 0, cols - 1)
        r = np.clip(np.rint(pts[:, 1] / dy).astype(np.int64), 0, rows - 1)
        return np.unique(r * cols + c)

    if min_dist is not None:
        idx = run(min_dist)
    else:
        min_dist = np.sqrt(width * height / n_target)
        idx = run(min_dist)
        for _ in range(calib_iters):
            min_dist *= np.sqrt(len(idx) / n_target)
            idx = run(min_dist)
    feats, idx = _features(xv, yv, elev, idx)
    return feats, idx, min_dist
