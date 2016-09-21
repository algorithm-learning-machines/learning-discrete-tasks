locales = {'en_US.UTF-8'}
os.setlocale(locales[1])

require("torch")
require("rnn")
require("optim")

require("header")

require("tasks.all_tasks")
require("models.memory_model")


function CustomModel()
   opt = {}
   opt.vectorSize = dataset.vectorSize
   opt.inputSize = dataset.inputSize
   local ShiftLearn = require('ShiftLearn')

   --------------------------------------------------------------------------------
   -- TODO should integrate these options nicely
   --------------------------------------------------------------------------------

   opt.separateValAddr = true 
   opt.noInput = true
   opt.noProb = true 
   opt.simplified = true 
   opt.supervised = true 

   if not opt.noProb then
      opt.maxForwardSteps = 5
   else
      opt.maxForwardSteps = dataset.repetitions
   end
   opt.epochs = 50 

   --------------------------------------------------------------------------------

   local model = Model.create(opt, ShiftLearn.createWrapper,
   ShiftLearn.createWrapper, nn.Identity, "modelName")

   --local model = Model.create(opt)


end

local opts = {}
opts.fixedLength = true

local tasks = allTasks()

for k,v in ipairs(tasks) do
   if v == "Copy" then
      local t = getTask(v)()
      
      local opt = {} 

      opt.memOnly = true
      opt.vectorSize = 5 
      opt.inputSize = 10 
      opt.separateValAddr = true 
      opt.noInput = true 
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
      local myModel = Model.create(opt)

      local seqLSTM = nn.Sequencer(myModel)
      X = torch.randn(2,1,5,5)
      out = seqLSTM:forward(X)
      print("main",out)
      
   end
end
