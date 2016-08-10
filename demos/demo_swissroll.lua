-- This is the Swissroll test...
local mani = require 'manifold'
local g = require 'gnuplot'

local function newFig(t) g.figure(); g.grid(true); g.title(t) end
local N = 2000
local K = 12
local sigma = 7
local d = 2

local tt = torch.mul(torch.sqrt(torch.rand(1, N)), 4 * math.pi)
local X = torch.DoubleTensor(3, N)
X[1] = torch.cmul(torch.add(tt, 0.1), torch.cos(tt))
X[2] = torch.cmul(torch.add(tt, 0.1), torch.sin(tt))
X[3] = torch.mul(torch.rand(1, N), 8 * math.pi)

newFig('Original 3D')
g.scatter3(X[1], X[2], X[3])

X = X:t()

print('random embedding...')
Y = mani.embedding.random(X, {
   dim = 2,
})

newFig('Random')
g.plot(Y, '+')

print('LLE embedding...')
Y = mani.embedding.lle(X, {
   dim = d,
   neighbors = K,
   tol = 1e-3
})

newFig('LLE')
g.plot(Y, '+')

print('Laplacian Eigenmaps embedding...')
local K = 60
Y = mani.embedding.laplacian_eigenmaps(X, {
   dim = 2,neighbors = K,sigma = sigma,normalized = false
})

newFig('Laplacian Eigenmaps')
g.plot(Y, '+')
