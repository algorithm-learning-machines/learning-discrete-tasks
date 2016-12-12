require 'rnn'
require("criterions.generic_criterion")
require("tasks.all_tasks")
-- [0.9869500000001] [704.70000000008]
-- TODO: twice as slow as keras version - do smth about this

os.setlocale('en_US.UTF-8')

local oldPrint = print
function myPrint(sender, ...)
   io.write(string.color("[" .. "LSTM TEST" .. "] ", "red"))
   io.write(string.color("[" .. sender .. "] ", "green"))
   oldPrint(...)
end

print = myPrint

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


local tasks = allTasks()
--------------------------------------------------------------------------------
-- Change options here to test stuff.

local opts = {}

opts.batchSize = 3
opts.positive = 1
opts.negative = -1
opts.trainMaxLength = 16
opts.testMaxLength = 20
opts.fixedLength = true
opts.onTheFly = true
opts.trainSize = 6000
opts.batchSize = 3000
opts.testSize = 3000
opts.verbose = true

-- Task specific options
opts.vectorSize = 10
opts.mean = 0.5
opts.maxCount = 5
opts.inputsNo = 3
learnRate = 0.001
beta1 = 0.9
beta2 = 0.999
epsilon = 0.00000001

local timer = torch.Timer()
-- For each task, train and test
for _, taskName in pairs(tasks) do

   print("MAIN", "Training model on " .. taskName)

   local Task = getTask(taskName)
   task = Task(opts)

   local inSize = 0
   local outSize = 0
   for k, v in pairs(task:getInputsInfo()) do
      inSize = inSize + v.size
   end
   local model = nn.Sequential():add(nn.LSTM(inSize, 16))
   local concatL = nn.Concat(1) -- concatenate separate output types
   local outputInfo = task:getOutputsInfo()
   for k, v in pairs(outputInfo) do
      outSize = outSize + v.size
      concatL:add(nn.GRU(16, v.size))
   end
   model:add(concatL)
   local criterion = GenericCriterion(task, opts)  -- create a generic criterion

   task:resetIndex("train")

   local stats = {loss={}}

   --while not task:isEpochOver() or (opts.onTheFly and i < 100) do

      X, T, F, L = task:updateBatch()
      -- to consider if concatenating input ok approach
      for i = 2, #X do
         X[1] = torch.cat(X[1],X[i])
      end
      Xt = X[1]
      timer:reset()
      t = 0
   for epoch=1,10 do
      e = 0

      for bt = 1,Xt:size()[2] do -- for every batch
         model:zeroGradParameters()
         model:forget()
         outSeq = {}
         for i = 1,opts.trainMaxLength do -- for every elem in sequence
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
               outSeq[i] = {Yt, Tt}
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
         for i = opts.trainMaxLength,1,-1 do -- reverse order of calls
               local loss = criterion:forward(outSeq[i][1], outSeq[i][2])
               local dYt = criterion:backward(outSeq[i][1], outSeq[i][2])
               for i = 2, #dYt do
                  dYt[1] = torch.cat(dYt[1],dYt[i])
               end
               -- momentum = (momentum or torch.zeros(dYt[1]:size())) * 0.9 + torch.pow(dYt[1],2) * 0.1
               -- dYt[1]:cdiv(torch.sqrt(momentum))
               momentum = (momentum or torch.zeros(dYt[1]:size())) * beta1 + dYt[1] * (1 - beta1)
               speed = (speed or torch.zeros(dYt[1]:size())) * beta2 + torch.pow(dYt[1],2) * (1 - beta2)
               momentum1 = torch.div(momentum, 1 - math.pow(beta1, t))
               speed1 = torch.div(speed, 1 - math.pow(beta2, t))
               momentum1:cdiv(torch.sqrt(speed1) + epsilon)
               model:backward(Xt[i][bt], momentum1)
               e = e + loss
         end
         model:updateParameters(learnRate)
         t = t + 1
      end

      --task:displayCurrentBatch()

      e = e/(Xt:size()[2] * opts.trainMaxLength)
      stats.loss[epoch] = e
      print("MAIN","Error after epoch "..epoch..": "..e)
      collectgarbage()
   end
   print("MAIN","Finished training in "..timer:time().real..' seconds')
   plot(stats,0)

   ---[[ Testing
   task:resetIndex("test")
   stats = {loss = 0, bitAcc = 0, totAcc = 0}
   --while not task:isEpochOver("test") or (opts.onTheFly and i > 0) do

      X, T, F, L = task:updateBatch("test")
      -- to consider if concatenating input ok approach
      for i = 2, #X do
         X[1] = torch.cat(X[1],X[i])
      end
      Xt = X[1]

      for bt = 1,Xt:size()[2] do -- for every batch
      	 model:zeroGradParameters()
      	 model:forget()
         for i = 1,opts.testMaxLength do -- for every elem in sequence
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
               stats.loss = stats.loss + evaluation.loss
               stats.bitAcc = stats.bitAcc + evaluation.correct/evaluation.n
               if evaluation.correct==evaluation.n then stats.totAcc = stats.totAcc + 1 end
            end
            if task:hasTargetAtTheEnd() and step == length then
               local loss = criterion:forward({Y}, {T[1][1][bt]})
               local dYt = criterion:backward({Y}, {T[1][1][bt]})
               evaluation = task:evaluateBatch({Y}, {T[1][1][bt]})[1]
               accurracy = accurracy + evaluation.correct/evaluation.n
            end
         end
      end

      --task:displayCurrentBatch()

      --sys.sleep(0.02)
      --i = i - 1
   --end--]]
   local testn = opts.testSize*opts.testMaxLength
   print("MAIN", "% of correct bits: "..tostring(stats.bitAcc/testn))
   print("MAIN", "% of correct answers: "..tostring(stats.totAcc/testn))
   print("MAIN", "Test loss: "..tostring(stats.loss/testn))

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
