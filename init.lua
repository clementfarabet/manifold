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
local distances = function(vectors,norm)
   -- args:
   local X = vectors
   local norm = norm or 2
   local N,D = X:size(1),X:size(2)

   -- compute L2 distances:
   local distances
   if norm == 2 then
      local X2 = X:clone():cmul(X):sum(2)
      distances = (X*X:t()*-2) + X2:expand(N,N) + X2:reshape(1,N):expand(N,N)
      distances:abs():sqrt()
   elseif norm == 1 then
      distances = X.new(N,N)
      local tmp = X.new(N,D)
      for i = 1,N do
         local x = X[i]:clone():reshape(1,D):expand(N,D)
         tmp[{}] = X
         local dist = tmp:add(-1,x):abs():sum(2):squeeze()
         distances[i] = dist
      end
   else
      error('norm must be 1 or 2')
   end
   
   -- return dists
   return distances
end

-- Compute neighbors:
local neighbors = function(vectors,norm)
   -- args:
   local X = vectors
   local N,D = X:size(1),X:size(2)

   -- compute L2 distances:
   local distance = distances(X,norm)
   
   -- sort:
   local dists,index = distance:sort(2)

   -- insure identity for 1st index:
   for i = 1,(#distance)[1] do
      local id1 = index[{i,1}]
      if id1 ~= i then
         for j = 2,(#distance)[1] do
            local id2 = index[{i,j}]
            if id2 == i then
               index[{i,j}] = id1
               index[{i,1}] = id2
               break
            end
         end
      end
   end

   -- return index
   return index,dists
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
   local matrix = torch.Tensor(#oks,D)
   for i,ok in ipairs(oks) do
      matrix[i] = X[ok]
   end

   -- return new filtered matrix:
   return matrix,oks
end

-- Package:
return {
   embedding = {
      random = random,
      lle = require 'manifold.lle',
      tsne = require 'manifold.tsne',
      laplacian_eigenmaps = require 'manifold.laplacian_eigenmaps',
   },
   removeDuplicates = removeDuplicates,
   neighbors = neighbors,
   distances = distances,
}
