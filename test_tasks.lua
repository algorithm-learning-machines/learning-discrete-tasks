--------------------------------------------------------------------------------
-- After implementing a new task, test it here.
--------------------------------------------------------------------------------

locales = {'en_US.UTF-8'}
os.setlocale(locales[1])

local createDumbModel = require("models.dumb_model")
local memoryModel = require("models.memory_model")
require("models.memory_model_wrapper")
require("models.sequencer_experiment")

require("criterions.generic_criterion")
require("criterions.mem_criterion")

--------------------------------------------------------------------------------
-- Add tasks here to test them.
--------------------------------------------------------------------------------

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
opt.allOutputs = false

--------------------------------------------------------------------------------
-- Simulate a training process with a memory model 

for _, T in pairs(tasks) do                                    -- take each task
   t = T(opt)
   if (t.name == "Copy") then
      local seq = SequencerExp(5, memoryModel.createMyModel, t, opt)  
      local m = createDumbModel(t, opt)                   -- create a dumb model
      local m1 = memoryModelWrapper(t, opt)
      local c = MemCriterion(t, opt)           -- create a generic criterion
      c.noAsserts = true                              
      t:resetIndex("train")
      local i = 0
      local err = {}
      while not t:isEpochOver() or (opt.onTheFly and i < 100) do
         X, T, F, L = t:updateBatch()
         local l = L[1]              -- whole batch has the same sequence length
         for s = 1, l do                              -- go through the sequence
            local Xt, Tt = {}, {}
            for k,v in pairs(X) do Xt[k] = v[s] end

            -- Problem -> memory model cannot process batches in parallel
            -- Problem -> memory model is not tailored for direct inputs
            
            -- Yt is memory1]
            local X_l = Xt[1]:size(1)
            local mem = torch.Tensor(opt.memorySize - X_l, Xt[1]:size(2))
            mem = torch.cat(Xt[1], mem, 1)
             
            local dummyAddress = torch.Tensor(opt.memorySize) -- temporary
            input_table = {mem, dummyAddress}
            
            local Yt = seq:forwardSequence(input_table)[1]
            print(Yt:size())
            -- t:evaluateBatch(Yt, Xt, err)

            if t:hasTargetAtEachStep() then
               for k,v in pairs(T) do Tt[k] = v[s] end
               local loss = c:forward(Yt, Tt[1])
               local dYt = c:backward(Yt, Tt[1])
               local dOutputs_mem = torch.Tensor(opt.memorySize -
                  X_l, Xt[1]:size(2))

               local dYt_mem = torch.cat(dYt, dOutputs_mem, 1)
               seq:backward(input_table, {dYt_mem, torch.Tensor(opt.memorySize)})
            end

            if t:hasTargetAtTheEnd() and s == l then
               for k,v in pairs(T) do Tt[k] = v[1] end

               local loss = c:forward(Yt, Tt[1])
               local dYt = c:backward(Yt, Tt[1])
               local dOutputs_mem = torch.Tensor(opt.memorySize -
                  X_l, Xt[1]:size(2))

               local dYt_mem = torch.cat(dYt, dOutputs_mem, 1)

               seq:backward(input_table, {dYt_mem, torch.Tensor(opt.memorySize)})

            end
         end

         -- t:displayCurrentBatch()
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
      --for s = 1, l do                               -- go through the sequence
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
