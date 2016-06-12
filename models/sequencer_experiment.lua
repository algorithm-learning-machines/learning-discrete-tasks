--------------------------------------------------------------------------------
-- Load needed modules
--------------------------------------------------------------------------------

require("nn")
require("nngraph")


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


--------------------------------------------------------------------------------
-- SequencerExp
--------------------------------------------------------------------------------


local SequencerExp = torch.class("SequencerExp")

function SequencerExp:__init(length, UnitModel, task, opt)

   opt = opt or {}
   self.noAsserts = opt.noAsserts or false
   self.verbose = opt.verbose or false

   if not self.noAsserts then
      assert(tonumber(length) > 0)
   end

   self.length = tonumber(length)
   self.batchSize = opt.batchSize == nil and 1 or opt.batchSize
   self.verbose = opt.verbose or false
   self.allOutputs = opt.allOutputs or true

   self.clones = {}

   self.unitModel = UnitModel(task, opt)

   self.clones[1] = self.unitModel 
   self.params, self.gradParams = self.clones[1]:getParameters()

   local params, gradParams = self.clones[1]:parameters()
   for j = 2, self.length do
      self.clones[j] = cloneModel(self.unitModel)
   end

   if self.verbose then
      self:message("Initialized SequencerExp with " .. self.length .. " clones.")
   end

   self:reset()

end

function SequencerExp:getParameters()
   return self.params, self.gradParams
end

function SequencerExp:parameters()
   return self.clones[1]:parameters()
end

function SequencerExp:zeroGradParameters()
   self.clones[1]:zeroGradParameters()
end

function SequencerExp:reset()
   self.crtIdx = 1
   self.crtLength = 0
end

function SequencerExp:forward(input, isTraining)
   isTraining = isTraining == nil and true or isTraining
   assert(self.crtIdx <= self.length or isTraining == false)

   if isTraining then
      if self.crtIdx == 1 then
         self.clones[self.crtIdx]:forward(input)
      else
         self.clones[self.crtIdx]:forward(input)
      end
      self.crtLength = self.crtLength + 1
      self.crtIdx = self.crtIdx + 1
      return self.clones[self.crtIdx-1].output
   else
      if self.crtLength == 0 then
         self.clones[self.crtIdx]:forward(input)
      else
         self.clones[1]:forward(input)
      end
      self.crtLength = self.crtLength + 1
      return self.clones[1].output
   end
end

function SequencerExp:forwardSequence(input, isTraining)
   local length = self.length or input:size(1)
   isTraining = isTraining == nil and true or isTraining
   self:reset()
   local y = {} 
   y[0] = input
   --print(input)
   --print("gigie")
   for i = 1, length do
      if (not isTraining) and self.allOutputs then
         y[i] = self:forward(y[i - 1], isTraining):clone()
      else
         y[i] = self:forward(y[i - 1], isTraining)
      end
   end

   if isTraining then
      assert((self.crtIdx == length + 1) and (self.crtLength == length))
   else
      assert((self.crtIdx == 1) and (self.crtLength == length))
   end
   
   self.y = y
   self.allOutputs = false 
   if self.allOutputs then
      return y
   else
      print(y[length])
      return y[length]
   end

end

function SequencerExp:backward(input, gradOutput)
   self.crtIdx = self.crtIdx - 1                    -- go back to the last clone
   local outGrad, inSignal
   if self.crtIdx == self.crtLength then
      outGrad = {gradOutput}
   else
      if gradOutput then
         outGrad = {
            self.clones[self.crtIdx+1].gradInput[2] + gradOutput
         }
      else
         outGrad = {
            self.clones[self.crtIdx+1].gradInput[2]
         }
      end
   end

   self.clones[self.crtIdx]:backward(inSignal, {outGrad[1], torch.Tensor(20)})
   return self.clones[self.crtIdx].gradInput
end

function SequencerExp:backwardSequence(input, gradOutput)
   local dx = {}
   if type(gradOutputs) == "table" then
      for i = self.crtLength, 1, -1 do
         dx[i] = self:backward(self.y[i + 1], gradOutput[i])
      end
   else
      dx[self.crtLength] = self:backward(inputs[self.crtLength], gradOutput)
      for i = self.crtLength-1, 1, -1 do
         dx[i] = self:backward(sef.y[i])
      end
   end
   assert(self:backpropOver())
   return dx
end

function SequencerExp:backpropOver()
   return self.crtIdx == 1
end


function SequencerExp:message(message)
   if self.verbose then
      print(
         string.format("[LSTM Seq] "):color("blue") ..
            string.format(message):color("green")
      )
   end
end

function SequencerExp:cuda()
   require("cutorch")
   require("cunn")

   self.clones[1]:cuda()
   self.params, self.gradParams = self.clones[1]:getParameters()

   local params, gradParams = self.clones[1]:parameters()
   for j = 2, self.length do
      self.clones[j]:cuda()
      if params and gradParams then
         local crtParams, crtGradParams = self.clones[j]:parameters()
         if not self.noAsserts then
            assert((#crtParams == #params) and (#crtGradParams == #gradParams))
         end
         for i = 1, #params do
            crtParams[i]:set(params[i])
         end
         for i = 1, #gradParams do
            crtGradParams[i]:set(gradParams[i])
         end
      end
   end

end

function SequencerExp:showParameters(zoom)
   self.unitModel:showParameters(self.clones[1], zoom)
end

