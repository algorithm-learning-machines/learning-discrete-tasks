require "nn"
require "rnn"
require "nngraph"
local memory_model = require("models.memory_model")

local memoryModelWrapper =  torch.class("memoryModelWrapper") 

function memoryModelWrapper:__init(task, opt)
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


function memoryModelWrapper:forward(X, l, isTraining)
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
