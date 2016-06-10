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

--------------------------------------------------------------------------------
-- helper function for returning final cloned module
--------------------------------------------------------------------------------

function cloneModel(model)
   local params, gradParams
   if model.parameters then
      params, gradParams = model:parameters()
      if params == nil then
         params = {}
      end
   end
   local paramsNoGrad
   if model.parametersNoGrad then
      paramsNoGrad = model:parametersNoGrad()
   end
   local mem = torch.MemoryFile("w"):binary()
   mem:writeObject(model)

   local reader = torch.MemoryFile(mem:storage(), "r"):binary()
   local clone = reader:readObject()
   reader:close()

   if model.parameters then
      local cloneParams, cloneGradParams = clone:parameters()
      local cloneParamsNoGrad
      for i = 1, #params do
         --Sets reference to model's parameters
         cloneParams[i]:set(params[i])
         cloneGradParams[i]:set(gradParams[i])

      end
      if paramsNoGrad then
         cloneParamsNoGrad = clone:parametersNoGrad()
         for i =1,#paramsNoGrad do
            ---- Sets reference to model's parameters
            cloneParamsNoGrad[i]:set(paramsNoGrad[i])
         end
      end
   end
   collectgarbage()
   mem:close()
   return clone
end


function memoryModelWrapper:cloneModel()
  newModel = memoryModel(self.task, self.opt)
  newModel.model = cloneModel(self.model)
  return newModel
end

