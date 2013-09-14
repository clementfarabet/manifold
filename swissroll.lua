require 'torch'
mani = require 'manifold_local'

local N = 2000
local K = 12
local d = 2

local tt = torch.mul(torch.add(torch.mul(torch.rand(1, N), 2), 1), (3*3.1416/2))
local height = torch.mul(torch.rand(1, N), 21)
local X = torch.DoubleTensor(3, N)
X[1] = torch.cmul(tt, torch.cos(tt))
X[2] = height
X[3] = torch.cmul(tt, torch.sin(tt))
--X[1] = torch.div(torch.linspace(1, N, N), N) --torch.cmul(tt, torch.cos(tt))
--X[2] = torch.div(torch.linspace(1, N, N), N) --height
--X[3] = torch.div(torch.linspace(1, N, N), N) --torch.cmul(tt, torch.sin(tt))
X = X:t()
local opts = {}
opts.dim = d
opts.neighbors = K
opts.tol = 1e-3

Y = mani.embedding.lle(X, opts)
