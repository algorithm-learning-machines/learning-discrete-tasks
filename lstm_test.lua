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

--print = myPrint

local function plot(series, tpause)
   local gp = io.popen("gnuplot", 'w')

   local lines = { }
   for k,v in pairs(series) do
      table.insert(lines, string.format(" '-' u 1:2 title '%s'", k))
   end
   
   gp:write("plot" .. table.concat(lines, ",") .. " with linespoints\n")
   for k,v in pairs(series) do
      for i=1,#v do
    gp:write(string.format("%f %f\n", i, v[i]))
      end
      gp:write("e\n")
   end

   gp:write(string.format("pause %f\n", tpause or 100.0))
   gp:close()
end


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
opts.trainSize = 1500
opts.testSize = 90
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
   local model = nn.Sequential():add(nn.GRU(inSize, 10)) -- hopefully this does the trick
   --model:add(nn.LSTM(10,10))
   local concatL = nn.Concat(1)
   local outputInfo = task:getOutputsInfo()
   for k, v in pairs(outputInfo) do
      outSize = outSize + v.size
      concatL:add(nn.GRU(10, v.size))
   end
   model:add(concatL)
   local criterion = GenericCriterion(task, opts)  -- create a generic criterion

   task:resetIndex("train")

   local i = 0
   local err = {}
   --err['error'] = {}

   while not task:isEpochOver() or (opts.onTheFly and i < 100) do

      X, T, F, L = task:updateBatch()
      -- to consider if concatenating input ok approach
      for i = 2, #X do
         X[1] = torch.cat(X[1],X[i])
      end
      Xt = X[1]
      e = 0

      for bt = 1,Xt:size()[2] do -- for every batch
         for i = 1,L[1] do -- for every elem in sequence
            local Y = model:forward(Xt[i][bt])

            if task:hasTargetAtEachStep() then
               Tt = {}
               Yt = {}
               for k,v in pairs(T) do
                  Tt[k] = v[i][bt]
               end
               j = 1
               for k,v in pairs(outputInfo) do
                  l = v.size
                  Yt[k] = Y:narrow(1, j, l)
                  j = j + l
               end
               local loss = criterion:forward(Yt, Tt)
               local dYt = criterion:backward(Yt, Tt)
               for i = 2, #dYt do
                  dYt[1] = torch.cat(dYt[1],dYt[i])
               end
               model:zeroGradParameters()
               model:backward(Xt[i][bt], dYt[1])
               model:updateParameters(0.01)--learning rate
               model:maxParamNorm(2)
               --print(dYt[1]:sum())
               e = e + dYt[1]:abs():sum()
            end
            if task:hasTargetAtTheEnd() and step == length then
               local loss = criterion:forward({Y}, {T[1][1][bt]})
               local dYt = criterion:backward({Y}, {T[1][1][bt]})
               model:zeroGradParameters()
               model:backward(Xt[i][bt], dYt[1])
               model:updateParameters(0.01)--learning rate
               model:maxParamNorm(2)
               --print(dYt[1]:sum())
               e = e + dYt[1]:abs():sum()
            end
         end
      end

      --task:displayCurrentBatch()

      --sys.sleep(0.02)
      i = i + 1
      --err["error"][i] = e
      print(e)
   end
   --plot(err)

   ---[[ Testing
   task:resetIndex("test")
   print("Testing...")
   accurracy = 0
   accn = 0
   while not task:isEpochOver("test") or (opts.onTheFly and i > 0) do

      X, T, F, L = task:updateBatch("test")
      -- to consider if concatenating input ok approach
      for i = 2, #X do
         X[1] = torch.cat(X[1],X[i])
      end
      Xt = X[1]

      for bt = 1,Xt:size()[2] do -- for every batch
         for i = 1,L[1] do -- for every elem in sequence
            local Y = model:forward(Xt[i][bt])

            if task:hasTargetAtEachStep() then
               Tt = {}
               Yt = {}
               for k,v in pairs(T) do
                  Tt[k] = v[i][bt]
               end
               j = 1
               for k,v in pairs(outputInfo) do
                  l = v.size
                  Yt[k] = Y:narrow(1, j, l)
                  j = j + l
               end
               evaluation = task:evaluateBatch(Yt, Tt)[1]
               accurracy = accurracy + evaluation.correct/evaluation.n
               accn = accn + 1
            end
            if task:hasTargetAtTheEnd() and step == length then
               local loss = criterion:forward({Y}, {T[1][1][bt]})
               local dYt = criterion:backward({Y}, {T[1][1][bt]})
               evaluation = task:evaluateBatch({Y}, {T[1][1][bt]})[1]
               accurracy = accurracy + evaluation.correct/evaluation.n
               accn = accn + 1
            end
         end
      end

      --task:displayCurrentBatch()

      --sys.sleep(0.02)
      --i = i - 1
   end--]]
   print(accurracy/accn)

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
