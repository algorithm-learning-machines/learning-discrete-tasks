locales = {'en_US.UTF-8'}
os.setlocale(locales[1])

require("torch")
require("rnn")
require("optim")

require("header")

require("tasks.all_tasks")

local opts = {}
opts.fixedLength = true

local tasks = allTasks()

for k,v in ipairs(tasks) do
   if v == "Copy" then
      local t = getTask(v)()

      -- model to train
      local seqLSTM = nn.Sequencer(nn.LSTM(t.totalInSize, t.totalOutSize))
      local X, T, F, L = t:updateBatch()

      while true do -- epoch

         parameters, gradParameters = seqLSTM:getParameters() 

         local feval = function(x)
            if x ~= parameters then parameters:copy(x) end    -- get new parameters
            gradParameters:zero()                                -- reset gradients
            local f = 0
            local train_count = 0

            while not t:isEpochOver() and train_count < 30 do 
               local X, T, F, L = t:updateBatch()
               local out = seqLSTM:forward(X)
               local err = t:evaluateBatch(out, T)

               de = t.criterions[1]:backward(out, T[1])

               seqLSTM:backward(X, de)

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
