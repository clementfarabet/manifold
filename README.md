Manifold
========

A package to maniuplate manifolds, for Torch7.

Install
-------

```sh
torch-rocks install https://raw.github.com/clementfarabet/manifold/master/manifold-scm-0.rockspec
```

Use
---

```lua
torch
> m = require 'manifold'
> t = torch.randn(100,10) -- 100 samples, 10-dim each
> p = m.embedding.lle(t, {dim=2})  -- embed samples into a 2D plane, using random projections
> p = m.embedding.lle(t, {dim=2, neighbors=3})  -- embed samples into a 2D plane, using 3 neighbor (LLE)
> ns = m.neighbors(t) -- return the matrix of neighbors for all samples (sorted)
> ds = m.distances(t) -- return the matrix of distances (L2)
```
