locales = {'en_US.UTF-8'}
os.setlocale(locales[1])

require("torch")
require("rnn")
require("optim")

require("header")

require("tasks.all_tasks")

require("models.memory_model")


function CustomModel()
   local opt = {} 

   opt.memOnly = true
   opt.vectorSize = 5 
   opt.inputSize = 10 
   opt.separateValAddr = true 
   opt.noInput = false -- model receives input besides its memory 
   opt.noProb = true 
   opt.simplified = false 
   opt.supervised = false 
   opt.probabilityDiscount = 0.99
   opt.maxForwardSteps = 5
   opt.batchSize = 2
   opt.memorySize = 5
   opt.useCuda = false


   ---- model to train
   Model = require("models.memory_model")
   return Model.create(opt)
end

local useOurModel = false

local tasks = allTasks()

 

for k,v in ipairs(tasks) do
   if v == "Copy" then
      local t = getTask(v)()

      -- model to train
      local seqModel
      if useOurModel then
         -- desired usage: nn.CustomModel(t.totalInSize, t.totalOutSize)
         seqModel = nn.Sequencer(CustomModel())
      else
         seqModel = nn.Sequencer(nn.LSTM(t.totalInSize, t.totalOutSize))
      end
      local X, T, F, L = t:updateBatch()

      while true do -- epoch

         parameters, gradParameters = seqModel:getParameters() 

         local feval = function(x)
            if x ~= parameters then parameters:copy(x) end    -- get new parameters
            gradParameters:zero()                                -- reset gradients
            local f = 0
            local train_count = 0

            while not t:isEpochOver() and train_count < 30 do 
               local X, T, F, L = t:updateBatch()
               -- for the sake of seeing stuff work
               if useOurModel then
                  X = torch.randn(2,1,35)
                  T = torch.randn(2,1,5,5)
               end

               local out = seqModel:forward(X)
               err = t:evaluateBatch(out, T) -- not working for custom model usage
               de = t.criterions[1]:backward(out, T[1])

               seqModel:backward(X, de)

               f = f + err[1].loss
               train_count = train_count + 1
            end

            f = f / train_count
            gradParameters:div(train_count)

            print("main", gradParameters)

            return f, gradParameters
         end

         optim.asgd(feval, parameters, {}) -- optimizer
      end
   end
end
