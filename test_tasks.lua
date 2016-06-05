--------------------------------------------------------------------------------
-- After implementing a new task, test it here.
--------------------------------------------------------------------------------

locales = {'en_US.UTF-8'}
os.setlocale(locales[1])

local createDumbModel = require("models.dumb_model")
require("criterions.generic_criterion")

--------------------------------------------------------------------------------
-- Add tasks here to test them.

require("tasks.copy_first")
require("tasks.copy")
local tasks = {Copy, CopyFirst}

--------------------------------------------------------------------------------
-- Change options here to test stuff.

local opt = {}

opt.batchSize = 3
opt.positive = 1
opt.negative = 0
opt.trainMaxLength = 10
opt.testMaxLength = 20
opt.fixedLength = false
opt.onTheFly = true
opt.trainSize = 1000
opt.testSize = 1000
opt.verbose = true

-- Task specific options
opt.vectorSize = 10
opt.mean = 0.9

--------------------------------------------------------------------------------
-- Simulate a training process with a dumb model

for _, T in pairs(tasks) do                                    -- take each task
   t = T(opt)

   local m = createDumbModel(t, opt)                      -- create a dumb model
   local c = GenericCriterion(t, opt)              -- create a generic criterion

   t:resetIndex("train")
   local i = 0

   while not t:isEpochOver() and (opt.onTheFly and i < 100) do
      X, T, F, L = t:updateBatch()
      for t = 1, L[1] do                              -- go through the sequence
         local Xt, Tt = {}, {}
         for k,v in pairs(X) do Xt[k] = v[t] end
         for k,v in pairs(T) do Tt[k] = v[t] end
         local Yt = m:forward(Xt)
         local loss = c:forward(Yt, Tt)
         local dYt = c:backward(Yt, Tt)
         m:backward(Xt, dYt)
      end
      t:displayCurrentBatch()
      sys.sleep(0.02)
      i = i + 1
   end

   t:resetIndex("test")
   while not t:isEpochOver("test") and (opt.onTheFly and i > 0) do
      X, T, F, L = t:updateBatch("test")
      for t = 1, L[1] do                              -- go through the sequence
         local Xt, Tt = {}, {}
         for k,v in pairs(X) do Xt[k] = v[t] end
         for k,v in pairs(T) do Tt[k] = v[t] end
         local Yt = m:forward(Xt)
         local loss = c:forward(Yt, Tt)
         local dYt = c:backward(Yt, Tt)
         m:backward(Xt, dYt)
      end
      t:displayCurrentBatch("test")
      sys.sleep(0.02)
      i = i - 1
   end

end
