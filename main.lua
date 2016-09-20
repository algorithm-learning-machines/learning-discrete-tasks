locales = {'en_US.UTF-8'}
os.setlocale(locales[1])

require("torch")
require("rnn")

require("header")

require("tasks.all_tasks")

local opts = {}
opts.fixedLength = true

local tasks = allTasks()

for k,v in ipairs(tasks) do
   if v == "Copy" then
      local t = getTask(v)()
      local X, T, F, L = t:updateBatch()

      local seqLSTM = nn.Sequencer(nn.LSTM(t.totalInSize, t.totalOutSize))
      local out = seqLSTM:forward(X)
      print("main", "Forward phase throuh LSTM is over")
      local err = t:evaluateBatch(out, T)
      print("main", "Batch evaluation is over")
   end
end
