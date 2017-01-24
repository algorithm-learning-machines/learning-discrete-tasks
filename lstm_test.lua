require 'rnn'
require 'optim'
require("criterions.generic_criterion")
require("tasks.all_tasks")

os.setlocale('en_US.UTF-8')

local oldPrint = print
function myPrint(sender, ...)
   io.write(string.color("[" .. "LSTM TEST" .. "] ", "red"))
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

local function trainPass(X, T, epoch, stats)
	model:zeroGradParameters()
	local Y = model:forward(X)
	local loss = criterion:forward(Y, T[1])
	local gradOut = criterion:backward(Y, T[1])
	gradOut = adamLR(gradOut, {})
	model:backward(X, gradOut)
	model:updateParameters(1) -- lr already incorporated into gradient
	eval = task:evaluateBatch(Y, T[1])[1]
	print("MAIN","Loss after epoch "..epoch..": "..eval.loss)
    stats['loss'][epoch] = eval.loss
    stats['acc'][epoch] = eval.correct / eval.n
end

local function testPass(X, T)
	model:zeroGradParameters()
	local Y = model:forward(X)
	eval = task:evaluateBatch(Y, T[1])[1]
	return eval.loss, eval.correct/eval.n
end

function adamLR(gradient, state)
   local state = state or {}
   local lr = state.learningRate or 0.001
   local lrd = state.learningRateDecay or 0

   local beta1 = state.beta1 or 0.9
   local beta2 = state.beta2 or 0.999
   local epsilon = state.epsilon or 1e-8
   local wd = state.weightDecay or 0

   -- (2) weight decay
   -- if wd ~= 0 then
   --    gradient:add(wd, x)
   -- end

   -- Initialization
   state.t = state.t or 0
   -- Exponential moving average of gradient values
   state.m = state.m or torch.zeros(gradient:size())
   -- Exponential moving average of squared gradient values
   state.v = state.v or torch.zeros(gradient:size())
   -- A tmp tensor to hold the sqrt(v) + epsilon
   state.denom = state.denom or torch.zeros(gradient:size())

   -- (3) learning rate decay (annealing)
   local clr = lr / (1 + state.t*lrd)

   state.t = state.t + 1

   -- Decay the first and second moment running average coefficient
   state.m:mul(beta1):add(1-beta1, gradient)
   state.v:mul(beta2):addcmul(1-beta2, gradient, gradient)

   state.denom:copy(state.v):sqrt():add(epsilon)

   local biasCorrection1 = 1 - beta1^state.t
   local biasCorrection2 = 1 - beta2^state.t
   local stepSize = clr * math.sqrt(biasCorrection2)/biasCorrection1
   gradient = torch.mul(torch.cdiv(state.m, state.denom), stepSize)

   return gradient
end

local tasks = allTasks()
--------------------------------------------------------------------------------
-- Change options here to test stuff.

local opts = {}

opts.positive = 1
opts.negative = -1
opts.trainMaxLength = 16
opts.testMaxLength = 20
opts.fixedLength = true
opts.onTheFly = true
opts.trainSize = 6000
opts.batchSize = 300
opts.testSize = 3000
opts.verbose = true

-- Task specific options
opts.vectorSize = 10
opts.mean = 0.5
opts.maxCount = 5
opts.inputsNo = 3

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
   model = nn.Sequential():add(nn.LSTM(inSize, 16))
   local concatL = nn.Concat(1) -- concatenate separate output types
   local outputInfo = task:getOutputsInfo()
   for k, v in pairs(outputInfo) do
      outSize = outSize + v.size
      concatL:add(nn.LSTM(16, v.size))
   end
   model:add(concatL)
   model = nn.Sequencer(model)
   -- local criterion = GenericCriterion(task, opts)  -- create a generic criterion
   criterion = nn.SequencerCriterion(nn.MSECriterion())

   task:resetIndex("train")

   local stats = {loss={},acc={},valid_loss={},valid_acc={}}

   --while not task:isEpochOver() or (opts.onTheFly and i < 100) do

      X, T, F, L = task:updateBatch()
      valX, valT, F, L = task:updateBatch()
      -- to consider if concatenating input ok approach
      for i = 2, #X do
         X[1] = torch.cat(X[1],X[i])
         valX[1] = torch.cat(valX[1],valX[i])
      end
      timer:reset()
   for epoch=1,20 do
      trainPass(X[1], T, epoch, stats)
      loss, bitsCorr = testPass(valX[1], valT)
      stats['valid_loss'][epoch] = loss
      stats['valid_acc'][epoch] = bitsCorr
      collectgarbage()
   end
   print("MAIN","Finished training in "..timer:time().real..' seconds')
   plot(stats,10)

   ---[[ Testing
   task:resetIndex("test")
   -- --while not task:isEpochOver("test") or (opts.onTheFly and i > 0) do

      X, T, F, L = task:updateBatch("test")
      -- to consider if concatenating input ok approach
      for i = 2, #X do
         X[1] = torch.cat(X[1],X[i])
      end
      loss, bitsCorr = testPass(X[1], T)
	  print("MAIN","Test loss: "..loss)
	  print("MAIN","Test bit accuracy: "..bitsCorr)

   --    for bt = 1,Xt:size()[2] do -- for every batch
   --    	 model:zeroGradParameters()
   --    	 model:forget()
   --       for i = 1,opts.testMaxLength do -- for every elem in sequence
   --          local Y = model:forward(Xt[i][bt])

   --          if task:hasTargetAtEachStep() then
   --             Tt = {}
   --             Yt = {}
   --             for k,v in pairs(T) do
   --                Tt[k] = v[i][bt]
   --             end
   --             j = 1
   --             for k,v in pairs(outputInfo) do
   --                l = v.size
   --                Yt[k] = Y:narrow(1, j, l)
   --                j = j + l
   --             end
   --             evaluation = task:evaluateBatch(Yt, Tt)[1]
   --             stats.loss = stats.loss + evaluation.loss
   --             stats.bitAcc = stats.bitAcc + evaluation.correct/evaluation.n
   --             if evaluation.correct==evaluation.n then stats.totAcc = stats.totAcc + 1 end
   --          end
   --          if task:hasTargetAtTheEnd() and step == length then
   --             local loss = criterion:forward({Y}, {T[1][1][bt]})
   --             local dYt = criterion:backward({Y}, {T[1][1][bt]})
   --             evaluation = task:evaluateBatch({Y}, {T[1][1][bt]})[1]
   --             accurracy = accurracy + evaluation.correct/evaluation.n
   --          end
   --       end
   --    end

   --    --task:displayCurrentBatch()

   --    --sys.sleep(0.02)
   --    --i = i - 1
   -- --end--]]
   -- local testn = opts.testSize*opts.testMaxLength
   -- print("MAIN", "% of correct bits: "..tostring(stats.bitAcc/testn))
   -- print("MAIN", "% of correct answers: "..tostring(stats.totAcc/testn))
   -- print("MAIN", "Test loss: "..tostring(stats.loss/testn))

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
