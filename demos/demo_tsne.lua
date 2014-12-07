
local manifold = require 'manifold'

-- function to show an MNIST 2D group scatter plot:
local function show_scatter_plot(mapped_x, labels)

  -- count label sizes:
  local K = 10
  local cnts = torch.zeros(K)
  for n = 1,labels:nElement() do
    cnts[labels[n] + 1] = cnts[labels[n] + 1] + 1
  end

  -- separate mapped data per label:
  mapped_data = {}
  for k = 1,K do
    mapped_data[k] = { key = 'Digit ' .. k - 1, values = torch.Tensor(cnts[k], opts.ndims) }
  end
  local offset = torch.Tensor(K):fill(1)
  for n = 1,labels:nElement() do
    mapped_data[labels[n] + 1].values[offset[labels[n] + 1]]:copy(mapped_x[n])
    offset[labels[n] + 1] = offset[labels[n] + 1] + 1
  end

  -- show results in scatter plot:
  local gfx = require 'gfx.js'
  gfx.chart(mapped_data, {
     chart = 'scatter',
     width = 600,
     height = 600,
  })
end


-- show map with original digits:
local function show_map(mapped_data, X)
  
  -- draw map with original digits:
  local im_size = 2048
  local background = 0
  local background_removal = true
  local map_im = manifold.draw_image_map(mapped_data, X:resize(X:size(1), 1, 28, 28), im_size, background, background_removal)
  
  -- plot results:
  local gfx = require 'gfx.js'
  gfx.image(map_im)
end


-- function that performs demo of t-SNE code on MNIST:
local function demo_tsne()

  -- amount of data to use for test:
  local N = 5000

  -- load subset of MNIST test data:
  local mnist = require 'mnist'
  local testset = mnist.testdataset()
  testset.size  = N
  testset.data  = testset.data:narrow(1, 1, N)
  testset.label = testset.label:narrow(1, 1, N)
  local x = torch.Tensor(testset.data:size())
  x:map(testset.data, function(xx, yy) return yy end)
  x:resize(x:size(1), x:size(2) * x:size(3))
  local labels = testset.label

  -- run t-SNE:
  local timer = torch.Timer()
  opts = {ndims = 2, perplexity = 30, pca = 100, use_bh = false}
  mapped_x1 = manifold.embedding.tsne(x, opts)
  print('Successfully performed t-SNE in ' .. timer:time().real .. ' seconds.')
  show_scatter_plot(mapped_x1, labels)
  show_map(mapped_x1, x:clone())

  -- run Barnes-Hut t-SNE:
  opts = {ndims = 2, perplexity = 30, pca = 100, use_bh = true, theta = 0.5}
  timer:reset()
  mapped_x2 = manifold.embedding.tsne(x, opts)
  print('Successfully performed Barnes Hut t-SNE in ' .. timer:time().real .. ' seconds.')
  show_scatter_plot(mapped_x2, labels)
  show_map(mapped_x2, x:clone())
end


-- run the demo:
demo_tsne()

