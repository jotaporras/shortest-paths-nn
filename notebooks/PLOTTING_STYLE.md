# Transferability Plot Style Spec

Camera-ready style for "test accuracy vs. training fraction" line plots used in the *Transferability of Graph Transformers via Manifolds* NeurIPS 2026 submission. Drop into any repo to render figures that read as part of the same paper.

This is **Style 1 (Minimal Editorial)** from `notebooks/2026-05-04-27-neurips_plot_styles.ipynb`. It is the default — alternative palettes are listed at the bottom.

## Locked colours

These three must match exactly across repos so the same model reads the same way in every figure:

| Model      | Hex       | Role     |
| ---------- | --------- | -------- |
| GNN        | `#7f8c8d` | baseline |
| Dense GT   | `#3d5a80` | proposed |
| Sparse GT  | `#81b29a` | proposed |

If the other repo adds more models, extend from the same family:

| Extra model | Hex       |
| ----------- | --------- |
| Exphormer   | `#e07a5f` |
| GraphGPS    | `#f2cc8f` |

Don't reuse `#3d5a80` or `#81b29a` for non-proposed methods.

## Markers (legend decodable without colour)

| Model     | Marker |
| --------- | ------ |
| GNN       | `o`    |
| Dense GT  | `D`    |
| Sparse GT | `v`    |
| Exphormer | `s`    |
| GraphGPS  | `^`    |

White marker edges (0.5 px) on every model so adjacent points don't blur.

## Line treatment (visual hierarchy)

| Class                         | Linestyle | LW factor | Marker factor | Alpha | z-order |
| ----------------------------- | --------- | --------- | ------------- | ----- | ------- |
| GNN (baseline)                | `--`      | 0.55      | 0.78          | 0.78  | 2       |
| Other transformer baselines   | `-`       | 1.00      | 1.00          | 0.95  | 3       |
| Proposed (Dense GT, Sparse GT)| `-`       | 1.05      | 1.05          | 1.00  | 5       |

Base linewidth ≈ 1.2, base markersize ≈ 3.5 — sized for in-paper rendering, not for screen previews.

GNN is dashed and thinner so it reads as a *reference* line; the proposed methods sit on top.

## Typography & axes (NeurIPS 2026 conventions)

NeurIPS 2026 body text is 10-pt Times Roman. Plot text matches the body so figures read as part of the prose.

- **Font family**: `Times New Roman → Times → DejaVu Serif` (serif).
- **Math**: `mathtext.fontset = "stix"` so `$\alpha$`, `$\mathbb{R}^d$` blend with Times. For exact LaTeX rendering, set `text.usetex = True` and `font.serif = ["Times"]`.
- **Sizes**: body 9, axis labels 9, panel titles 9 (left-aligned), ticks/legend 8. Designed to render at actual textwidth without rescaling.
- **PDF embedding**: `pdf.fonttype = 42`, `ps.fonttype = 42`, `svg.fonttype = "none"`. NeurIPS rejects PDFs with Type 3 fonts; 42 is TrueType.
- **Figure size**: render at NeurIPS single-column textwidth (~5.5 in). Include with `\includegraphics{file.pdf}` and **do not rescale** — `[width=\textwidth]` distorts effective font size relative to body text.
- No top/right spines. Left/bottom spines + ticks recoloured `#444`.
- X label: `r"Training fraction $\alpha$"`; Y label: `"Test accuracy"`.
- Layout: `constrained_layout=True`, 2×2 grid at `figsize=(5.5, 4.0)`. Single bottom-centre legend, `ncol=len(models)`, `frameon=False`.

## Drop-in code

Self-contained — copy into a notebook or `paper_plots.py`:

```python
import matplotlib as mpl
import matplotlib.pyplot as plt

PALETTE = {
    "GNN":       "#7f8c8d",
    "Exphormer": "#e07a5f",
    "GraphGPS":  "#f2cc8f",
    "Dense GT":  "#3d5a80",
    "Sparse GT": "#81b29a",
}
MARKERS = {
    "GNN":       "o",
    "Exphormer": "s",
    "GraphGPS":  "^",
    "Dense GT":  "D",
    "Sparse GT": "v",
}
BASELINES = {"GNN"}
PROPOSED  = {"Dense GT", "Sparse GT"}

def line_attrs(model, base_lw=1.2, base_ms=3.5):
    if model in BASELINES:
        return dict(lw=base_lw * 0.55, ms=base_ms * 0.78,
                    alpha=0.78, ls="--", zorder=2)
    if model in PROPOSED:
        return dict(lw=base_lw * 1.05, ms=base_ms * 1.05,
                    alpha=1.0, ls="-", zorder=5)
    return dict(lw=base_lw, ms=base_ms, alpha=0.95, ls="-", zorder=3)

RC = {
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "svg.fonttype":      "none",
    "axes.unicode_minus": False,
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset":  "stix",
    "font.size":         9,
    "axes.labelsize":    9,
    "axes.titlesize":    9,
    "axes.titlepad":     4,
    "axes.titlelocation": "left",
    "legend.fontsize":   8,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "axes.linewidth":    0.7,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
}


def plot_transferability(df, datasets, model_order,
                         dataset_titles=None,
                         x_col="train_frac", y_col="test_acc",
                         model_col="Model", dataset_col="dataset",
                         out_path="transferability.pdf"):
    """
    df : long-form DataFrame with columns [model_col, dataset_col, x_col, y_col].
    datasets : list of dataset keys, plotting order.
    model_order : list of model names matching PALETTE / MARKERS keys; sets legend order.
    dataset_titles : dict[str, str] mapping dataset key -> panel title.
    """
    dataset_titles = dataset_titles or {d: d for d in datasets}
    n = len(datasets)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    with mpl.rc_context(RC):
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5, 2.0 * nrows),
                                 constrained_layout=True, squeeze=False)
        for ax, dskey in zip(axes.ravel(), datasets):
            sub = df[df[dataset_col] == dskey]
            for model in model_order:
                g = sub[sub[model_col] == model].sort_values(x_col)
                if g.empty:
                    continue
                ax.plot(g[x_col], g[y_col],
                        marker=MARKERS[model], color=PALETTE[model], label=model,
                        markeredgecolor="white", markeredgewidth=0.5,
                        **line_attrs(model))
            ax.set_title(dataset_titles[dskey])
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.spines["left"].set_color("#444")
            ax.spines["bottom"].set_color("#444")
            ax.tick_params(colors="#444")
        for ax in axes[-1, :]:
            ax.set_xlabel(r"Training fraction $\alpha$")
        for ax in axes[:, 0]:
            ax.set_ylabel("Test accuracy")
        for ax in axes.ravel()[len(datasets):]:
            ax.set_visible(False)
        h, l = axes.ravel()[0].get_legend_handles_labels()
        order = [l.index(m) for m in model_order if m in l]
        fig.legend([h[i] for i in order], [l[i] for i in order],
                   loc="lower center", ncol=len(model_order), frameon=False,
                   bbox_to_anchor=(0.5, -0.04))
        fig.savefig(out_path, bbox_inches="tight")
        plt.show()
```

### Minimal usage

```python
import pandas as pd

df = pd.DataFrame({
    "Model":      [...],   # one of: "GNN", "Dense GT", "Sparse GT", ...
    "dataset":    [...],   # dataset key
    "train_frac": [...],   # 0..1
    "test_acc":   [...],   # 0..1
})

plot_transferability(
    df,
    datasets=["arxiv-year", "ogbn-mag", "snap-patents"],
    model_order=["GNN", "Dense GT", "Sparse GT"],
    dataset_titles={
        "arxiv-year":   "Arxiv-Year (76.5K nodes)",
        "ogbn-mag":     "OGBN-MAG (855K nodes)",
        "snap-patents": "SNAP-Patents (1.71M nodes)",
    },
    out_path="transferability.pdf",
)
```

## NeurIPS PDF font check

NeurIPS rejects camera-ready submissions whose figures contain Type 3 fonts. The `pdf.fonttype = 42` setting above produces TrueType. To verify the final compiled paper:

```bash
pdffonts paper.pdf | grep -E "Type 3|type3"   # must return nothing
```

If you use `text.usetex = True`, double-check that your TeX install isn't bringing back Type 3 via `cm-super` / Computer Modern — pin Times-Roman in `font.serif` and let `mathtext.fontset = "stix"` handle math.

## Adapting to other model rosters

- **Subset only.** If the other repo only has GNN + Dense GT + Sparse GT, pass `model_order=["GNN", "Dense GT", "Sparse GT"]` and the rest is automatic. The locked colours will match figures from this repo.
- **Renaming.** If the repo's column has different display names (e.g., `"sparse-gt"` instead of `"Sparse GT"`), remap before plotting — don't change `PALETTE` keys, since the colour-to-role binding is what makes the figures align.
- **New baselines.** Add a new entry to `PALETTE` + `MARKERS`; treat them as "other transformer baselines" (default weight via `line_attrs`).
- **New proposed methods.** Add to `PROPOSED` so they get the heavier line treatment, and pick a hue distinct from `#3d5a80` / `#81b29a`.

## Single-row variant (teaser figure)

For an introduction/teaser figure that spans full text width, swap the 2×2 grid for `1×N` and use `figsize=(5.5, 1.4)`. Everything else (palette, markers, line treatment, fonts) stays. Reduce x-axis tick density to `MaxNLocator(4)`. This is **Style 3** in the source notebook.

## Alternative palettes

The locked editorial palette is the default. If you need to swap, keep the GNN-as-grey-baseline convention. Recommended substitutes (all tested in `notebooks/2026-05-04-27-neurips_plot_styles.ipynb`):

- **Okabe-Ito** (colourblind-safe): GNN `#999999`, Dense GT `#0072B2`, Sparse GT `#009E73`, Exphormer `#E69F00`, GraphGPS `#56B4E9`.
- **Tableau-10**: GNN `#9ca3af`, Dense GT `#4E79A7`, Sparse GT `#59A14F`, Exphormer `#F28E2B`, GraphGPS `#76B7B2`.
- **seaborn:colorblind** with GNN forced to `#7f8c8d`.

Whichever palette you commit to, fix it across repos — switching mid-paper makes reviewers think there are two different models.
