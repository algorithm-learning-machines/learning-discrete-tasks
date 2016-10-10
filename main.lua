locales = {'en_US.UTF-8'}
os.setlocale(locales[1])

require("torch")
require("rnn")
require("optim")

require("header")

require("tasks.all_tasks")

require("models.memory_model")
require("models.memory_model_wrapper")

function CustomModel()
   local opt = {} 

   opt.memOnly = true
   opt.vectorSize = 5 
   opt.inputSize = 10 
   opt.separateValAddr = true 
   opt.noInput = false -- model receives input besides its memory 
   --opt.outputLine = true
   opt.noProb = true 
   opt.simplified = false 
   opt.supervised = false 
   opt.probabilityDiscount = 0.99
   opt.maxForwardSteps = 5
   opt.batchSize = 2
   opt.memorySize = 5
   opt.useCuda = false

   -- model to train
   Model = require("models.memory_model")
   return Model.create(opt)
end


function CustomModelWrapper(i, o)
   local opt = {} 

   opt.memOnly = true
   opt.vectorSize = o 
   opt.inputSize = i-- should be the same as vectorSize 
   opt.separateValAddr = true 
   opt.noInput = false -- model receives input besides its memory 
   opt.outputLine = true
   opt.noProb = true 
   opt.simplified = false 
   opt.supervised = false 
   opt.probabilityDiscount = 0.99
   opt.maxForwardSteps = 5
   opt.batchSize = 2
   opt.memorySize = 5 -- number of columns practically
   opt.useCuda = false
   model = memoryModelWrapper(opt)--= require("models.memory_model_wrapper")

   -- model to train
   return model 

end

local cmd = torch.CmdLine() 
cmd:text()
cmd:option('-useOurModel', true, 'Use our custom model')
cmd:text()

local opts = cmd:parse(arg or {})

local tasks = allTasks()


-- small example to show Sequencer works with rnn LSTM
local model = nn.Sequencer(nn.LSTM(10,10))
model:forward(torch.Tensor(10,1,10))
model:backward(torch.Tensor(10,1,10), torch.Tensor(10,1,10))
-- folowing model does not work!
-- local modelOne = CustomModelWrapper(10,10)
-- modelOne:forward(torch.Tensor(20,1,10))
-- modelOne:backward(torch.Tensor(20,1,10), torch.Tensor(20,1,10))

os.exit(0)

for k,v in ipairs(tasks) do
   if v == "Copy" then
      local t = getTask(v)()

      -- model to train
      local seqModel
      if opts.useOurModel then
         -- desired usage: nn.CustomModel(t.totalInSize, t.totalOutSize)
         seqModel = nn.Sequencer(CustomModelWrapper(t.totalInSize, t.totalOutSize))
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
               local err, out
               
               out = seqModel:forward(X)
               err = t:evaluateBatch(out, T)
               de = t.criterions[1]:backward(out, T[1])
               -- upper case should collapse to this as well in the end

               --print("de", tostring(de:size()))
               --print("X", tostring(X:size()))
               --print("sizes", tostring(t.totalInSize).." "..tostring(t.totalOutSize))

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
