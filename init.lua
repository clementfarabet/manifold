-- Deps:
require 'torch'

-- Random projections:
local random = function(vectors,opts)
   -- args:
   opts = opts or {}
   local dim = opts.dim or 2

   -- random mapping:
   local mapping = torch.randn(vectors:size(2),dim):div(dim)

   -- project:
   return vectors * mapping
end

-- Compute distances:
local distances = function(vectors)
   -- args:
   local X = vectors
   local N,D = X:size(1),X:size(2)

   -- compute L2 distances:
   local X2 = X:clone():cmul(X):sum(2)
   local distances = (X*X:t()*-2) + X2:expand(N,N) + X2:reshape(1,N):expand(N,N)
   distances:sqrt()
   
   -- return dists
   return distances
end

-- Compute neighbors:
local neighbors = function(vectors)
   -- args:
   local X = vectors
   local N,D = X:size(1),X:size(2)

   -- compute L2 distances:
   local X2 = X:clone():cmul(X):sum(2)
   local distance = (X*X:t()*-2) + X2:expand(N,N) + X2:reshape(1,N):expand(N,N)
   
   -- sort:
   local _,index = distance:sort(2)

   -- return index
   return index
end

-- Remove duplicates:
local removeDuplicates = function(vectors)
   -- args:
   local X = vectors
   local N,D = X:size(1),X:size(2)

   -- remove duplicates
   local neighbors = neighbors(X)

   -- mark single vectors as ok:
   local oks = {}
   for i = 1,N do
      if neighbors[i][1] == i then
         table.insert(oks,i)
      end
   end

   -- keep singles:
   local matrix = torch.FloatTensor(#oks,D)
   for i,ok in ipairs(oks) do
      matrix[i] = X[ok]
   end

   -- return new filtered matrix:
   return matrix,oks
end

-- LLE:
local lle = function(vectors,opts)
   -- args:
   opts = opts or {}
   local d = opts.dim or 2
   local K = opts.neighbors or 2
   local dtol = opts.tol or 2
   local X = vectors

   -- dims:
   local N,D = X:size(1),X:size(2)
   
   -- get nearest neighbors:
   local neighbors = neighbors(X)
   assert(torch.dist(neighbors[{{},1}]:float(), torch.range(1,N):float()) == 0, 'LLE cannot deal with duplicates')
   local neighborhood = neighbors[{ {},{2,2+K-1} }]
   
   -- solve for reconstruction weights:
   local tol = dtol or 0
   if K > D then
      tol = dtol or 1e-3 -- regularization in this case
   end
   local W = torch.zeros(N,K)
   local neighbors = torch.zeros(K,D)
   for ii = 1,N do
      -- copy neighbors:
      local indexes = neighborhood[ii]
      for i = 1,indexes:size(1) do
         neighbors[i] = X[indexes[i]]
      end

      -- shift point to origin:
      local z = neighbors - X[{ {ii,ii},{} }]:clone():expand(K,D)

      -- local covariance matrix:
      local C = z * z:t()

      -- regularize
      if tol > 0 then
         C:add( torch.eye(K)*tol*torch.trace(C) )
      end

      -- solve C*W=1
      local right = torch.ones(K,1)
      local res = torch.gels(right,C)
      W[ii] = res
      W[ii]:div(W[ii]:sum())
   end

   -- compute embedding from eigenvectors of cost matrix M = (I-W)' * (I-W)
   local M = torch.eye(N)
   for ii = 1,N do
      local w = W[ii]
      local indexes = neighborhood[ii]
      for i = 1,indexes:size(1) do
         local jj = indexes[i]
         M[{ {ii},{jj} }]:add(-w[i])
         M[{ {jj},{ii} }]:add(-w[i])
         M[{ {jj},{indexes[i]} }]:add( w[i]^2 )
      end
   end

   -- embedding:
   local vals,vectors = torch.symeig(M)
   local res = vectors[{ {2,2+d-1},{} }]:clone():t()
   res:mul(math.sqrt(N))

   -- return:
   return res
end

-- Package:
return {
   embedding = {
      lle = lle,
      random = random,
   },
   removeDuplicates = removeDuplicates,
   neighbors = neighbors,
   distances = distances,
}
