require "nn"
require "rnn"
require "nngraph"
require("../header")


local memory_model = require("models.memory_model")

local memoryModelWrapper =  torch.class("memoryModelWrapper", "nn.Module") 

function memoryModelWrapper:__init(opt)
   self.opt = opt or {}

   self.vectorSize = tonumber(opt.vectorSize)
   self.memSize = tonumber(opt.memorySize)

   self.model = nn.LSTM(self.memSize * self.vectorSize + self.vectorSize,
      self.memSize * self.vectorSize + self.vectorSize)
   --memory_model.create(self.opt)
   --self.mem = torch.Tensor(1, self.memSize, self.vectorSize) 
   self.mem = torch.Tensor(1, self.memSize * self.vectorSize)
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
function memoryModelWrapper:updateOutput(X)
   -- x is just one instance
   --print("sizes mem", self.memSize * self.vectorSize)
   --print("sizes X", X:size())
   local inp = torch.cat(self.mem:reshape(1,self.memSize * self.vectorSize), X) 
   
   self.inp = inp
   self.output = self.model:forward(inp)

   self.old_mem = self.mem
   self.mem = self.output:narrow(2,1,self.memSize * self.vectorSize)
   --self.mem = self.output[1]
   --self.output = self.output[2]
   return self.output:narrow(2,self.memSize * self.vectorSize, self.vectorSize)
end

function memoryModelWrapper:updateGradInput(X, dOutputs)
   --local inp = torch.cat(self.old_mem:reshape(1, self.memSize * self.vectorSize), X)
   local inp = torch.cat(self.mem, X)
   local dOutputs_mem = torch.zeros(1,self.memSize * self.vectorSize) -- temp, tudor
   
   print("derivate mem", tostring(dOutputs_mem:size()))
   print("derivate iesire", tostring(dOutputs:size()))
    
   self.gradInput = self.model:backward(inp, torch.cat(dOutputs_mem, dOutputs))
   print("gigi are mere", tostring(self.gradInput:size()))

   return self.gradInput 
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

