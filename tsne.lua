
-- implementation of P-value computations:
local function x2p(data, perplexity, tol)

  -- allocate all the memory we need:
  local eps = 1e-10
  local N = data:size(1)
  local D     = torch.DoubleTensor(N, N)
  local y_buf = torch.DoubleTensor(data:size())
  local row_D = torch.DoubleTensor(N, 1)
  local row_P = torch.DoubleTensor(N, 1)
  local P = torch.DoubleTensor(N, N)

  -- compute pairwise distance matrix:
  torch.cmul(y_buf, data, data)
  torch.sum(row_D, y_buf, 2)
  local row_Dc = torch.expand(row_D, N, N)
  torch.mm(D, data, data:t())
  D:mul(-2):add(row_Dc):add(row_Dc:t())

  -- loop over all instances:
  for n = 1,N do
  
    -- set minimum and maximum values for precision:
    local beta = 1
    local betamin = -math.huge
    local betamax =  math.huge

    -- compute the Gaussian kernel and corresponding perplexity: (should put this in a local function!)
    row_P:copy(D[n])
    row_D:copy(D[n])
    row_P:mul(-beta):exp()
    row_P[n][1] = 0
    local sum_P  = row_P:sum()
    local sum_DP = row_D:cmul(row_P):sum()
    local Hi = math.log(sum_P) + beta * sum_DP / sum_P

    -- evaluate whether the perplexity is within tolerance
    local H_diff = Hi - math.log(perplexity)
    local tries = 0
    while math.abs(H_diff) > tol and tries < 50 do

      -- if not, increase of decrease precision:
      if H_diff > 0 then
        betamin = beta
        if betamax == math.huge then
          beta = beta * 2
        else
          beta = (beta + betamax) / 2
        end
      else
        betamax = beta
        if betamin == -math.huge then
          beta = beta / 2
        else
          beta = (beta + betamin) / 2
        end
      end

      -- recompute row of P and correponding perplexity:
      row_P:copy(D[n])
      row_D:copy(D[n])
      row_P:mul(-beta):exp()
      row_P[n][1] = 0
      sum_P  = row_P:sum()
      sum_DP = row_D:cmul(row_P):sum()
      Hi = math.log(sum_P) + beta * sum_DP / sum_P
      
      -- update error:
      H_diff = Hi - math.log(perplexity)
      tries = tries + 1
    end

    -- set the final row of P:
    row_P:div(sum_P)
    P[n]:copy(row_P)
  end

  -- return output:
  return P
end


-- implementation of sign function:
local function sign(x)
  if x < 0 then
    return -1
  else
    return 1
  end
end


-- function that runs the Barnes-Hut SNE executable:
local function run_bhtsne(data, opts)

   -- default values:
   local ffi = require 'ffi'
   opts = opts or {}
   local no_dims    = opts.dim        or 2
   local perplexity = opts.perplexity or 30
   local theta      = opts.theta      or 0.5

   -- pack data:
   local raw = data:data()
   local nchars = data:nElement()*8 + 2*8 + 3*4
   local data_char = ffi.new('char[?]', nchars)
   local data_int = ffi.cast('int *', data_char)
   local data_double = ffi.cast('double *', data_char + 2*4)
   local data_int2 = ffi.cast('int *', data_char + 2*4 + 2*8)
   local data_double2 = ffi.cast('double *', data_char + 3*4 + 2*8)
   data_int[0] = data:size(1)
   data_int[1] = data:size(2)
   data_double[0] = ffi.cast('double',theta)
   data_double[1] = ffi.cast('double',perplexity)
   data_int2[0] = no_dims
   ffi.copy(data_double2, raw, data:nElement() * 8)

   -- pack argument
   local packed = ffi.string(data_char, nchars)
   local f = io.open('data.dat','w')
   f:write(packed)
   f:close()

   -- exec:
   local cmd
   -- TODO: we should prepend the exact install path to thee binaries
   if ffi.os == 'OSX' then
      cmd = 'bhtsne_maci'
   else
      cmd = 'bhtsne_linux'
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
      odata = torch.DoubleTensor(n,nd):zero()
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
      costs = torch.DoubleTensor(n):zero()
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

  -- return mapped data:
  return odata
end


-- implementation of t-SNE in Torch:
local function tsne(data, opts)

  -- options:
  opts = opts or {}
  local no_dims    = opts.dim        or 2
  local perplexity = opts.perplexity or 30
  local use_bh     = opts.use_bh     or false
  local pca_dims   = opts.pca        or (use_bh and 100) or nil

  -- normalize input data:
  data:add(-data:min())
  data:div( data:max())

  -- first do PCA:
  local N = data:size(1)
  local D = data:size(2)
  if pca_dims and pca_dims < D then
    require 'unsup'
    print('Performing preprocessing using PCA...')
    local lambda,W = unsup.pca(data)
    W = W:narrow(2, 1, pca_dims)
    data = torch.mm(data, W)
  end

  -- run Barnes-Hut binary (when requested):
  if use_bh == true then
    local mapped_x = run_bhtsne(data, opts)
    return mapped_x
  end

  -- initialize some variables for the optimization:
  local momentum = 0.5
  local final_momentum = 0.8
  local mom_switch_iter = 250
  local stop_lying_iter = 200
  local max_iter = 1000
  local epsilon = 500
  local min_gain = 0.01
  local eps = 1e-12

  -- allocate all the memory we need:
  local buf   = torch.DoubleTensor(N, N)
  local num   = torch.DoubleTensor(N, N)
  local Q     = torch.DoubleTensor(N, N)
  local y_buf = torch.DoubleTensor(N, no_dims)
  local n_buf = torch.DoubleTensor(N, 1)

  -- compute (asymmetric) P-values:
  print('Computing P-values...')
  local tol = 1e-4
  local P = x2p(data, perplexity, tol)

  -- symmetrize P-values:
  buf:copy(P)
  P:add(buf:t())
  P:div(torch.sum(P))

  -- compute constant term in KL divergence:
  buf:copy(P)
  buf:add(eps):log()
  local H_P = torch.sum(buf:cmul(P))

  -- lie about the P-values:
  P:mul(4)

  -- initialize the solution, gradient and momentum storage, and gain:
  local y_data = torch.randn(N, no_dims):mul(0.0001)
  local y_grad = torch.DoubleTensor(N, no_dims)
  local y_incs = torch.zeros(N, no_dims)
  local y_gain = torch.ones(N, no_dims)
 
  -- main for-loop:
  print('Running t-SNE...')
  for iter = 1,max_iter do
    
    -- compute the joint probability that i and j are neighbors in the map:
    torch.cmul(y_buf, y_data, y_data)
    torch.sum(n_buf, y_buf, 2) 
    local cp_n_buf = torch.expand(n_buf, N, N)
    torch.mm(num, y_data, y_data:t())
    num:mul(-2)
    num:add(cp_n_buf):add(cp_n_buf:t()):add(1)
    buf:fill(1)
    torch.cdiv(num, buf, num)
    for n = 1,N do
      num[n][n] = 0
    end
    torch.div(Q, num, num:sum())

    -- compute the gradients:
    buf:copy(P)
    buf:add(-Q)
    buf:cmul(num)
    torch.sum(n_buf, buf, 2)
    num:fill(0)
    for n = 1,N do
      num[n][n] = n_buf[n][1]
    end
    num:add(-buf)
    torch.mm(y_grad, num, y_data)
    y_grad:mul(4)

    -- update the solution:
    y_gain:map2(y_grad, y_incs, function(gain, grad, incs) if sign(grad) ~= sign(incs) then return (gain + 0.2) else return math.min(gain * 0.8, .01) end end)
    y_incs:mul(momentum)
    y_incs:addcmul(-epsilon, y_grad, y_gain)
    y_data:add(y_incs)
    y_data:add(-torch.mean(y_data, 1):reshape(1, no_dims):expand(y_data:size()))

    -- update learning parameters if necessary:
    if iter == mom_switch_iter then
      momentum = final_momentum
    end
    if iter == stop_lying_iter then
      P:div(4)
    end

    -- print out progress:
    if math.fmod(iter, 10) == 0 then
      Q:add(eps):log()
      local kl = H_P - torch.sum(Q:cmul(P))
      print('Iteration ' .. iter .. ': KL divergence is ' .. kl)
    end
  end

  -- return output data:
  return y_data, kl
end

-- return function:
return tsne
