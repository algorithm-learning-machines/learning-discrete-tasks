--------------------------------------------------------------------------------
-- After implementing a new task, test it here.
--------------------------------------------------------------------------------

locales = {'en_US.UTF-8'}
os.setlocale(locales[1])

require("tasks.copy_first")

local tasks = {CopyFirst}

local opt = {}

opt.batchSize = 3
opt.positive = 1
opt.negative = -1
opt.trainMaxLength = 10
opt.testMaxLength = 20
opt.fixedLength = false
opt.onTheFly = true
opt.trainSize = 1000
opt.testSize = 1000
opt.verbose = true

opt.vectorSize = 10

for _, T in pairs(tasks) do
   local i = 0
   t = T(opt)
   t:resetIndex("train")
   while not t:isEpochOver() and i < 100 do
      t:updateBatch()
      t:displayCurrentBatch()
      sys.sleep(0.02)
      i = i + 1
   end
   t:resetIndex("test")
   while not t:isEpochOver("test") and i > 0 do
      t:updateBatch("test")
      t:displayCurrentBatch("test")
      sys.sleep(0.02)
      i = i - 1
   end

end
