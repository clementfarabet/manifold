-- This is the Swissroll test...
require 'torch'
mani = require 'manifold'
require 'gfx.js'

local N = 2000
local K = 12
local sigma = 7
local d = 2

local tt = torch.mul(torch.sqrt(torch.rand(1, N)), 4*3.1416)
local X = torch.DoubleTensor(3, N)
X[1] = torch.cmul(torch.add(tt, 0.1), torch.cos(tt))
X[2] = torch.cmul(torch.add(tt, 0.1), torch.sin(tt))
X[3] = torch.mul(torch.rand(1, N), 8*3.1416)
X = X:t()

print('random embedding...')
Y = mani.embedding.random(X, {
   dim = 2,
})

gfx.chart({values=Y, key='Random'}, {chart='scatter', width=1024, height=800})

print('LLE embedding...')
Y = mani.embedding.lle(X, {
   dim = d,
   neighbors = K,
   tol = 1e-3
})

gfx.chart({values=Y, key='LLE'}, {chart='scatter', width=1024, height=800})

print('Laplacian Eigenmaps embedding...')
local K = 60
Y = laplacian_eigenmaps(X, {
   dim = 2,neighbors = K,sigma = sigma,normalized = false
})

gfx.chart({values=Y, key='Laplacian Eigenmaps'}, {chart='scatter', width=1024, height=800})
