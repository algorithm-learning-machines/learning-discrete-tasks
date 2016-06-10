--------------------------------------------------------------------------------
-- After implementing a new task, test it here.
--------------------------------------------------------------------------------

locales = {'en_US.UTF-8'}
os.setlocale(locales[1])

local createDumbModel = require("models.dumb_model")
local memory_model = require("models.memory_model")

require("criterions.generic_criterion")

--------------------------------------------------------------------------------
-- Add tasks here to test them.

require("tasks.copy_first")
require("tasks.copy")
require("tasks.doom_clock")
require("tasks.indexing")
require("tasks.get_next")
require("tasks.binary_sum")
require("tasks.subtract_on_signal")

local tasks = {
   GetNext, Indexing, DoomClock, Copy, CopyFirst, SubtractOnSignal, BinarySum
}

--------------------------------------------------------------------------------
-- Change options here to test stuff.

local opt = {}

opt.batchSize = 1
opt.positive = 1
opt.negative = -1
opt.trainMaxLength = 10
opt.testMaxLength = 20
opt.fixedLength = false
opt.onTheFly = false
opt.trainSize = 100
opt.testSize = 100
opt.verbose = true

-- Task specific options
opt.vectorSize = 10
opt.mean = 0.5
opt.maxCount = 5
opt.inputsNo = 4
opt.memorySize = 20

--------------------------------------------------------------------------------
-- Simulate a training process with a memory model 

for _, T in pairs(tasks) do                                    -- take each task
   t = T(opt)
   if (t.name == "copy") then
   local m = createDumbModel(t, opt)                      -- create a dumb model
   local m1 = memory_model.createMyModel(t, opt)
   local c = GenericCriterion(t, opt)              -- create a generic criterion
   c.noAsserts = true                              -- temp fix
   t:resetIndex("train")
   local i = 0
   local err = {}
   while not t:isEpochOver() or (opt.onTheFly and i < 100) do
      X, T, F, L = t:updateBatch()
      local l = L[1]                 -- whole batch has the same sequence length
      for s = 1, l do                                 -- go through the sequence
         local Xt, Tt = {}, {}
         for k,v in pairs(X) do Xt[k] = v[s] end
         --print(#Xt)
         --print("gigi are mere")
         Xt = Xt[1] -- todo remove this hack
         -- Problem -> memory model cannot process batches in parallel
         -- Problem -> #Xt == 2 in here seems hardcoded, why?
         -- Problem -> memory model is not tailored for direct inputs
         -- Should need a wrapper to move inputs to memory or 
         -- add input line to memory;
         local mem = torch.Tensor(opt.memorySize - 1, Xt:size(2))

         mem = torch.cat(Xt, mem, 1)  
         dummy = torch.Tensor(opt.memorySize)
            
         --local mem = torch.Tensor(m1.memSize, Xt[1]:size(1))
         local Yt = m1:forward({mem, dummy})[1][1]

         -- t:evaluateBatch(Yt, Xt, err)

         if t:hasTargetAtEachStep() then
            for k,v in pairs(T) do Tt[k] = v[s] end
            Tt = Tt[1]
            local loss = c:forward(Yt, Tt)
            local dYt = c:backward(Yt, Tt)
            local dummy_back = torch.Tensor(opt.memorySize)
            local mem = torch.Tensor(opt.memorySize - 1, Xt:size(2))
            m1:backward({torch.cat(Xt, mem, 1), dummy_back},
               {dYt, torch.Tensor(opt.memorySize)})
         end

         if t:hasTargetAtTheEnd() and s == l then
            for k,v in pairs(T) do Tt[k] = v[1] end
            Tt = Tt[1]

            local loss = c:forward(Yt, Tt)
            local dYt = c:backward(Yt, Tt)
            local dYt_mem = torch.Tensor(opt.memorySize - 1, Xt:size(2))
            dYt_mem = torch.cat(dYt:reshape(1, dYt:size()[1]), dYt_mem, 1)
            local dummy_back = torch.Tensor(opt.memorySize)
            local mem = torch.Tensor(opt.memorySize - 1, Xt:size(2))
            mem = torch.cat(Xt, mem, 1)
            --print(dummy_back:size())
            m1:backward({mem, dummy_back},
               {dYt_mem, torch.Tensor(opt.memorySize)})
         end
       end

      --t:displayCurrentBatch()
      -- t:printCurrentBatch()
      sys.sleep(0.02)
      i = i + 1
   end

   require("cutorch")
   require("cunn")
   m:cuda()
   t:cuda()
   c:cuda()

   --t:resetIndex("test")
   --while not t:isEpochOver("test") or (opt.onTheFly and i > 0) do
      --X, T, F, L = t:updateBatch("test")
      --local l = L[1]
      --for s = 1, l do                                 -- go through the sequence
         --local Xt, Tt = {}, {}
         --for k,v in pairs(X) do Xt[k] = v[s] end
         --local Yt = m:forward(Xt)

         --if t:hasTargetAtEachStep() then
            --for k,v in pairs(T) do Tt[k] = v[s] end
            --local loss = c:forward(Yt, Tt)
            --local dYt = c:backward(Yt, Tt)
            --m:backward(Xt, dYt)
         --end

         --if t:hasTargetAtTheEnd() and s == l then
            --for k,v in pairs(T) do Tt[k] = v[1] end
            --local loss = c:forward(Yt, Tt)
            --local dYt = c:backward(Yt, Tt)
            --print(loss)
            --print(dYt)
            --m:backward(Xt, dYt)
         --end
      --end
      ---- t:displayCurrentBatch("test")
      ---- t:printCurrentBatch("test")
      --sys.sleep(0.02)
      --i = i - 1
   --end
   end -- end if task == copy
end
