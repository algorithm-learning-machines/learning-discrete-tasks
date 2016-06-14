require "nn"
require "rnn"
require "nngraph"
local memory_model = require("models.memory_model")

local memoryModelWrapper =  torch.class("memoryModelWrapper") 

function memoryModelWrapper:__init(task, opt)
   self.task = task
   self.opt = opt or {}
   self.model = memory_model.createMyModel(task, self.opt)
end

function memoryModelWrapper:getParameters()
   return self.model:getParameters()
end

function memoryModelWrapper:parameters()
   return self.model:parameters()
end

function memoryModelWrapper:zeroGradParameters()
   return self.model:zeroGradParameters()
end

-- chained param tells whether model shall be used singularly or
-- in a sequence of clones; clones imply that output of one model shall
-- be input of the next clone 
function memoryModelWrapper:forward(X, l)
   local X_l = X:size(1)
   local mem = torch.Tensor(self.opt.memorySize - X_l, X:size(2))
   mem = torch.cat(X, mem, 1)
  local dummyAddress = torch.Tensor(self.opt.memorySize) -- temporary
   self.output = self.model:forward({mem, dummyAddress})

   return self.output
end

function memoryModelWrapper:backward(X, dOutputs, l)
   local X_l = X:size(1)
   local dummy_back = torch.Tensor(self.opt.memorySize) -- temporary
   local mem = torch.Tensor(self.opt.memorySize - X_l, X:size(2))
   local temp = torch.cat(X, mem, 1)
   local dOutputs_mem = torch.Tensor(self.opt.memorySize - X_l, X:size(2))

   dOutputs_mem = torch.cat(dOutputs:reshape(1, dOutputs:size()[1]),
      dOutputs_mem, 1)

   self.model:backward({mem, dummyAddress},{dOutputs_mem, dummy_back})
end


function memoryModelWrapper:cuda()
   require("cunn")
   self.model:cuda()
end

--------------------------------------------------------------------------------
-- helper function for returning final cloned module
--------------------------------------------------------------------------------


function memoryModelWrapper:cloneModel()
  newModel = memoryModel(self.task, self.opt)
  newModel.model = newModel:cloneModel(self.model)
  return newModel
end

