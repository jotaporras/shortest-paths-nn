# Receptive field versus physical radius (discussion, 2026-10-13)

**Status:** discussion note. Section 1 is JP's sketch. Sections 2–6 are Claude's responses, based on this repo only. Claude has not read the paper, so nothing here is checked against the paper's theorems. The transferability question is deferred to the paper repo.

## 1. JP's sketch

Follow-up project: how the receptive field of the graph transformer, its hop count k, relates to the physical radius R that a prediction must cover.

Intuition. For a given resolution r, you need a larger k to cover the same radius. As n grows toward the manifold limit, the physical receptive field of a fixed k shrinks to a small region of the manifold. This is a weakness of the convergence analysis. It does not explain why the graph transformer works well on terrain shortest-path distances.

Proposed bound. For balls to cover the manifold, their radius must be of order (log n / n)^(1/d). If

k = a · (log n / n)^(1/d) · n / (some square-root term in 2r),

then the receptive field matches the physical radius. This assumes that a shortest-path prediction needs the receptive field to cover the whole path.

Second observation. Shortest-path distance is a decomposable task. Good local steps are enough to approximate the global shortest path.

From `notebooks/notes.md`, the three regimes:

1. Fix the range, increase the resolution (manifold convergence). k fixed, ε decreases, so the receptive field decreases.
2. Increase range and resolution together, density fixed (local weak convergence). The receptive field stays the same.
3. Fix the range, increase the resolution, and match the receptive field to the range. Then k must grow with the resolution.

The task decides which regime applies. Data and model let us control (r, k).

## 2. The bound, corrected

The intuition is right and is the standard ε-graph picture. The sketched formula mixes the random-sample setting with the grid setting. The clean form is k ≈ R / ε_n, where ε_n is the edge length of the graph.

- Covering radius. For n uniform samples on a d-dimensional manifold, the connectivity and covering radius scales as ε_n ≍ (log n / n)^(1/d). This part of the sketch is correct. The log factor guarantees that every ball of radius ε_n contains a sample, so hop paths exist everywhere.
- Hops to physical radius. One hop covers geodesic distance at most ε_n. With enough intermediate samples, which the covering guarantees, k hops reach geodesic distance ≈ k · ε_n. So the k-hop window is a geodesic ball of radius k · ε_n. To cover radius R:

  k ≈ R / ε_n ≍ R · (n / log n)^(1/d).

  The two-sided bound between hop distance times ε and geodesic distance is the Isomap theorem: Bernstein, de Silva, Langford and Tenenbaum, "Graph approximations to geodesics on embedded manifolds" (2000). Reference to verify before use.
- The sketched expression. (log n / n)^(1/d) · n = n^(1−1/d) · (log n)^(1/d). In d = 2 this is √(n log n). It has the right power of n only because 1 − 1/d = 1/d when d = 2, and the log should divide, not multiply. In d = 3 it gives n^(2/3) instead of n^(1/3).
- The √2 · r term. On the 8-neighbour grid with spacing h, a hop covers between h and √2 · h. The hop count to reach R lies between R / (√2 h) and R / h. This is the grid version of the same bound, with ε_n replaced by h and no log factor, because the grid is not a random sample. Use one version or the other, not both in one formula.
- Layers. The receptive field of the model is L · k hops, not k. The RPEARL encoder adds its own hops (`rpearl_num_layers` × `pe_k`).

## 3. Numbers for the current model

Config `configs/sparse-gt-data-norway.yml`: 6 layers of 5 hops, so 30 hops. Norway `full_test-004` spans 20 km on a 500 × 500 grid at res04, so the spacing is about 40 m. Spacings below are derived from that, not read from the data files.

| Resolution | Spacing | 30-hop radius | Mean pair distance |
| ---------- | ------- | ------------- | ------------------ |
| res04      | 40 m    | 1.2–1.7 km    | 12 km              |
| res20      | 200 m   | 6–8.5 km      | 12 km              |

At res04 the two endpoint windows see at most 3.4 km of a 12 km path. At res20 they cover the typical path. The old e29 results are flat from res04 to res20 (`2026-10-results-analysis.md`, section 1). This fits section 3 of that analysis: the model learned a near-linear function of the xyz input coordinates, and that function needs zero hops.

## 4. The "whole path" assumption is only half true

The node features contain xyz, so the straight-line part of the distance needs no receptive field. The receptive field matters only for the terrain correction. On Norway that correction is 2% on the median pair and 5–8% on the top decile (detour ratios in the results analysis, section 2). The question is not "can the window cover the path" but "can the window cover the part of the path where the detour happens".

## 5. Decomposability, made precise

Shortest-path distance satisfies the Bellman equation

d(x, y) = min over neighbours z of x of  w(x, z) + d(z, y).

In the manifold limit this is the eikonal equation, a local PDE in x. So the update rule is local and learnable with a small k. But the number of updates must scale with D / (k · ε_n) to propagate information from y to x, as in Bellman-Ford. A fixed-depth network cannot do this beyond its receptive field. For pairs with D > 2 · L · k · ε_n, the siamese model predicts from coordinates plus local terrain statistics around each endpoint. The looped transformer in `notes.md` is the natural fix: shared weights iterated T times give receptive field T · k · ε_n at constant parameter count.

Iteration counts for Norway with k = 5 and a 12 km pair:

| Resolution | Spacing | Steps |
| ---------- | ------- | ----- |
| res04      | 40 m    | 60    |
| res20      | 200 m   | 12    |

## 6. A direct test and the theory gap

Test. Bin the test error by pair distance D and mark D = 2 · L · k · h on the plot. If the error rises past that line, the receptive field is the limit. If the curve is flat, the model is not using the window. The e29 setup can be rerun with k from 1 to 5 at fixed resolution.

Theory gap. With k fixed and ε_n → 0, the k-hop attention converges to a local operator, while the target d(x, ·) is nonlocal. Regime 3 of section 1 is k_n = R / ε_n → ∞. The limit object is attention restricted to a geodesic ball of radius R on the manifold. The theorem to prove is convergence of k_n-hop graph attention to that ball-restricted manifold attention. The Isomap bound in section 2 supplies the geometric ingredient. Whether this fits the paper's existing convergence theorem is open until the theorem is checked.
