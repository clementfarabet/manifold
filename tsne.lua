-- Deps:
require 'unsup'
local ffi = require 'ffi'

-- Wrapper to the tSNE binaries
local function tsne(data, opts)
   -- options:
   opts = opts or {}
   local ndims = opts.ndims or 2
   local perplexity = opts.perplexity or 30
   local landmarks = opts.landmarks or 1
   local pca = opts.pca

   -- first do PCA, and keep first N dimensions:
   if pca then
      data = unsup.pca_whiten(data)
      data = data[{ {},{1,math.min(pca, data:size(2))} }]
   end

   -- pack data:
   local raw = data:data()
   local nchars = data:nElement()*8 + 2*8 + 3*4
   local data_char = ffi.new('char[?]', nchars)
   local data_int = ffi.cast('int *', data_char)
   local data_double = ffi.cast('double *', data_char + 3*4)
   data_int[0] = data:size(1)
   data_int[1] = data:size(2)
   data_int[2] = ndims
   data_double[0] = ffi.cast('double',perplexity)
   data_double[1] = ffi.cast('double',landmarks)
   ffi.copy(data_double+2, raw, data:nElement() * 8)

   -- pack argument
   local packed = ffi.string(data_char, nchars)
   local f = io.open('data.dat','w')
   f:write(packed)
   f:close()

   -- exec:
   local cmd
   if ffi.os == 'OSX' then
      cmd = 'tSNE_maci'
   else
      cmd = 'tSNE_linux'
   end

   -- run:
   os.execute(cmd)

   -- read result:
   local f = io.open('result.dat','r')
   local res = f:read('*all')
   local data_int = ffi.cast('int *', res)
   local n = data_int[0]
   local nd = data_int[1]
   local data_next = data_int + 2

   -- clear files:
   os.remove('data.dat')
   os.remove('result.dat')
  
   -- data?
   local odata,lm,costs
   if n == 0 then
      -- no data (error?)
      print('no data found in embedding')
      return
   else
      -- data:
      odata = torch.zeros(n,nd)
      local rp = odata:data()
      local data_double = ffi.cast('double *', data_next)
      ffi.copy(rp, data_double, n*nd*8)
      local data_next = data_double + n*nd

      -- next vector:
      lm = torch.IntTensor(n)
      local rp = lm:data()
      local data_int = ffi.cast('int *', data_next)
      ffi.copy(rp, data_int, n*4)
      local data_next = data_int + n
      lm:add(1)
      
      -- next vector:
      costs = torch.zeros(n)
      local rp = costs:data()
      local data_double = ffi.cast('double *', data_next)
      ffi.copy(rp, data_double, n*8)
   end

   -- re-order:
   if landmarks == 1 then
      local odatar = odata:clone():zero()
      for i = 1,lm:size(1) do
         odatar[lm[i]] = odata[i]
      end
      odata = odatar
   end

   -- return output data
   return odata,lm,costs
end

-- Test:
local function test()
   -- sample data from two high-dim gaussians:
   local t = torch.zeros(1000,10)
   t[{ {1,500} }]:normal(1000,0.001)
   t[{ {501,750} }]:normal(-1000,0.001)
   t[{ {751,1000},{1,5} }]:normal(-1000,0.001)
   t[{ {751,1000},{5,10} }]:normal(1000,0.001)

   -- tSNE:
   local res,lm = tsne(t, {pca=30})

   -- render result, using two colors:
   require 'gfx.js'
   gfx.chart({
      {values = res[{ {1,500} }]},
      {values = res[{ {501,750} }]},
      {values = res[{ {751,1000} }]},
   }, 
   {chart='scatter', width=900, height=700})
end

-- Return func:
return tsne
