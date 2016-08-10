--------------------------------------------------------------------------------
-- After implementing a new task, test it here.
--------------------------------------------------------------------------------

locales = {'en_US.UTF-8'}
os.setlocale(locales[1])

local oldPrint = print
function myPrint(sender, ...)
   io.write(string.color("[" .. "DUMB TEST" .. "] ", "red"))
   io.write(string.color("[" .. sender .. "] ", "green"))
   oldPrint(...)
end

print = myPrint


require 'rnn'
local createDumbModel = require("models.dumb_model")
require("criterions.generic_criterion")

--------------------------------------------------------------------------------
-- Add tasks here to test them.

require("tasks.all_tasks")

local tasks = allTasks()
--------------------------------------------------------------------------------
-- Change options here to test stuff.

local opts = {}

opts.batchSize = 3
opts.positive = 1
opts.negative = -1
opts.trainMaxLength = 10
opts.testMaxLength = 20
opts.fixedLength = false
opts.onTheFly = false
opts.trainSize = 100
opts.testSize = 100
opts.verbose = true

-- Task specific options
opts.vectorSize = 10
opts.mean = 0.5
opts.maxCount = 5
opts.inputsNo = 3

--------------------------------------------------------------------------------
-- Simulate a training process with a dumb model

for _, taskName in pairs(tasks) do                             -- take each task

   print("MAIN", "Testing " .. taskName)

   local Task = getTask(taskName)
   task = Task(opts)

   --local model = createDumbModel(task, opts)              -- create a dumb model
   local inSize = 0
   local outSize = 0
   for k, v in pairs(task:getInputsInfo()) do
      inSize = inSize + v.size
   end
   for k, v in pairs(task:getOutputsInfo()) do
      outSize = outSize + v.size
   end
   local model = nn.Sequential():add(nn.LSTM(inSize, outSize)) -- hopefully this does the trick
   local criterion = GenericCriterion(task, opts)  -- create a generic criterion

   task:resetIndex("train")

   local i = 0
   local err = {}

   while not task:isEpochOver() or (opts.onTheFly and i < 100) do

      X, T, F, L = task:updateBatch()

	  for k,Xt in pairs(X) do
         for i = 1,Xt:size()[1] do
            local Yt = model:forward(Xt[i])

         --if task:hasTargetAtEachStep() then
         --   for k,v in pairs(T) do Tt[k] = v[step] end
            local loss = criterion:forward({Yt}, {T[k][i]})
            local dYt = criterion:backward({Yt}, {T[k][i]})
            model:backward(Xt[i], dYt[1])
            print("Error",dYt[1]:sum())
         --end

         --[[if task:hasTargetAtTheEnd() and step == length then
            for k,v in pairs(T) do Tt[k] = v[1] end
            local loss = criterion:forward(Yt, Tt)
            local dYt = criterion:backward(Yt, Tt)
            model:backward(Xt, dYt)--]]
         end
       end

      --task:displayCurrentBatch()

      sys.sleep(0.02)
      i = i + 1
   end

   -----------------------------------------------------------------------------
   -- Let's try on cuda now for the test data set
   -----------------------------------------------------------------------------
--[[
   print("MAIN", "Moving to CUDA")

   require("cutorch")
   require("cunn")

   model:cuda()
   task:cuda()
   criterion:cuda()

   task:resetIndex("test")
   while not task:isEpochOver("test") or (opts.onTheFly and i > 0) do

      X, T, F, L = task:updateBatch("test")

      local length = L[1]
      for step = 1, length do                         -- go through the sequence

         local Xt, Tt = {}, {}
         for k,v in pairs(X) do Xt[k] = v[step] end
         local Yt = model:forward(Xt)

         if task:hasTargetAtEachStep() then

            for k,v in pairs(T) do Tt[k] = v[step] end
            local loss = criterion:forward(Yt, Tt)
            local dYt = criterion:backward(Yt, Tt)
            model:backward(Xt, dYt)

         elseif task:hasTargetAtTheEnd() and step == length then

            for k,v in pairs(T) do Tt[k] = v[1] end
            local loss = criterion:forward(Yt, Tt)
            local dYt = criterion:backward(Yt, Tt)
            model:backward(Xt, dYt)

         end
      end
      task:displayCurrentBatch("test")

      sys.sleep(0.02)
      i = i - 1
   end

   print("MAIN", "Done with " .. taskName)
--]]
end
