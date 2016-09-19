locales = {'en_US.UTF-8'}
os.setlocale(locales[1])

require("torch")


require("rnn")
require("tasks.doom_clock")
require("tasks.all_tasks")

local opts = {}
opts.fixedLength = true

local tasks = allTasks()
--print(tasks)

for k,v in ipairs(tasks) do
   local t = getTask(v)() 
   local X, T, F, L = t:updateBatch()

   local inputSize, outputSize
  

   local seqLSTM = nn.Sequencer(nn.FastLSTM(t.inputSize, t.outputSize, 
      backSteps))
end

--local dc = DoomClock()

--local X, T, F, L = dc:updateBatch()

--local backSteps = 3 -- arbitrary value

--local seqLSTM = nn.Sequencer(nn.FastLSTM(dc.inputSize, dc.outputSize, 
   --backSteps))

--local X, T, F, L = dc:updateBatch()

--local seq = {}
--local input = X[1]
--for i=1,dc.trainMaxLength do
   --seq[#seq + 1] = input[i]
--end




--print(seqLSTM:forward(seq))

