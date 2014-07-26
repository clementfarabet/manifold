-- Implementation of Laplacian Eigenmaps: 
local laplacian_eigenmaps = function(vectors,opts)

   -- args:
   opts = opts or {}
   local d = opts.dim or 2
   local K = opts.neighbors or 5
   local sigma = opts.sigma or 1
   local normalized = opts.normalized or false
   local X = vectors

   -- dependencies:
   local pkg = require 'manifold'

   -- dims:
   local N,D = X:size(1),X:size(2)
   
   -- get nearest neighbors:
   local neighbors,dists = pkg.neighbors(X)
   assert(torch.dist(neighbors[{{},1}]:float(), torch.range(1,N):float()) == 0, 'Laplacian Eigenmaps cannot deal with duplicates')
   local neighborhood = neighbors[{ {},{1,K+1} }]
   local kernel = dists[{ {},{1,K+1} }]  

   -- compute Gaussian kernel:
   kernel:cmul(kernel)
   kernel:div(-2 * sigma * sigma)
   kernel:exp()
   local L = torch.zeros(N, N)
   for n = 1,N do
      for k = 1,K+1 do
         L[n][neighborhood[n][k]] = kernel[n][k]
      end
   end
   L:map(L:t(), function(xx, yy) if xx > yy then return xx else return yy end end )

   -- compute unnormalized graph Laplacian:
   if normalized == false then
      local kernel_sums = torch.sum(kernel, 2)
      L:mul(-1)
      for n = 1,N do
         L[n][n] = kernel_sums[n]
      end

   -- compute normalized graph Laplacian:
   else
      local kernel_sums = torch.sum(kernel, 2)
      kernel_sums:pow(-.5)
      L:cmul(kernel_sums:expand(N, N))
      L:cmul(kernel_sums:t():expand(N, N)) 
      L:map(L:t(), function(xx, yy) if xx > yy then return xx else return yy end end )
   end

   -- compute embedding:
   local vals,vectors = torch.eig(L, 'V')
   vals = vals[{{},1}]
   if normalized == true then
      vals,idx = torch.sort(vals,vals:dim(),true)
   else
      vals,idx = torch.sort(vals)
   end 
   local res = torch.Tensor(N, d) 
   for i=1,d do
      res[{{},i}] = vectors[{ {},{idx[i+1]} }]:clone()
   end

   -- return:
   return res
end

-- Return func:
return laplacian_eigenmaps
