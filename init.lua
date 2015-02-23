-- Deps:
require 'torch'
require 'sys'


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


-- function that draws 2D map of images:
local function draw_image_map(X, images, inp_map_size, inp_background, inp_background_removal)
  
  -- input options:
  local map_size = inp_map_size or 512
  local background = inp_background or 0
  local background_removal = inp_background_removal or false
  
  -- check inputs are correct:
  local N = X:size(1)
  if X:nDimension() ~= 2 or X:size(2) ~= 2 then
    error('This function is designed to operate on 2D embeddings only.')
  end
  if (type(images) ~= 'table' or X:size(1) ~= #images) and (torch.isTensor(images) == false or images:nDimension() ~= 4 or images:size(1) ~= N) then
    error('Images should be specified as a list of filenames or as an NxCxHxW tensor.')
  end
  
  -- prepare some variables:
  local num_channels = 3
  if torch.isTensor(images) then
    num_channels = images:size(2)
  end
  local map_im = torch.DoubleTensor(num_channels, map_size, map_size):fill(background)
  X = X:add(-X:min(1):expand(N, 2))
  X = X:cdiv(X:max(1):expand(N, 2))
  
  -- fill map with images:
  for n = 1,N do
    
    -- get current image:
    local cur_im
    if type(images) == 'table' then
      cur_im = image.load(images[n])
    else
      cur_im = images[n]
    end
     
    -- place current image:
    local y_loc = 1 + math.floor(X[n][1] * (map_size - cur_im:size(2)))
    local x_loc = 1 + math.floor(X[n][2] * (map_size - cur_im:size(3)))
    if background_removal == false then     -- no background removal
      map_im:sub(1, num_channels, y_loc, y_loc + cur_im:size(2) - 1,
                                  x_loc, x_loc + cur_im:size(3) - 1):copy(cur_im)
    else                                    -- background removal
      for c = 1,num_channels do
        for h = 1,cur_im:size(2) do
          for w = 1,cur_im:size(3) do
            if cur_im[c][h][w] ~= background then
              map_im[c][y_loc + h - 1][x_loc + w - 1] = cur_im[c][h][w]
            end
          end
        end
      end     
    end
  end
  
  -- return map:  
  return map_im
end


-- function that draw text map (requires qtlua):
local function draw_text_map(X, words, inp_map_size, inp_font_size)
  -- NOTE: This function assumes vocabulary is indexed by words, values indicate the index of a word into X!
  
  -- input options:
  local map_size  = inp_map_size or 1024
  local font_size = inp_font_size or 9
  
  -- check inputs are correct:
  local N = X:size(1)
  if X:nDimension() ~= 2 or X:size(2) ~= 2 then
    error('This function is designed to operate on 2D embeddings only.')
  end
  local num_words = 0
  for _,_ in pairs(words) do
    num_words = num_words + 1
  end  
  if X:size(1) ~= num_words then
    error('Number of words should match the number of rows in X.')
  end
  
  -- normalize the embedding:
  X = X:add(-X:min(1):expand(N, 2))
  X = X:cdiv(X:max(1):expand(N, 2))
  
  -- prepare image for rendering:
  require 'image'
  require 'qtwidget'
  require 'qttorch'
  local win = qtwidget.newimage(map_size, map_size)
  win:setcolor{r = 1, g = 1, b = 1}
  win:rectangle(1, map_size, 1, map_size)
  
  --render the words:
  for key,val in pairs(words) do
    win:setcolor{r = 0, g = 0, b = 0}
    win:setfont(qt.QFont{serif = false, size = fontsize})
    win:moveto(math.floor(X[val][1] * map_size), math.floor(X[val][2] * map_size))
    win:show(key)
  end
  
  -- render to tensor:
  local map_im = win:image():toTensor()
  
  -- return text map:
  return map_im
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
   draw_image_map = draw_image_map,
   draw_text_map  = draw_text_map
}
