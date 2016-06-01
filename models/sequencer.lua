--------------------------------------------------------------------------------
-- Load needed modules
--------------------------------------------------------------------------------

require("nn")
require("nngraph")

--------------------------------------------------------------------------------
-- Sequencer
--------------------------------------------------------------------------------

local Sequencer = torch.class("Sequencer")

function Sequencer:__init(length, inSize, outSize, UnitModel, opt)

   opt = opt or {}

   self.noAsserts = opt.noAsserts or false
   self.verbose = opt.verbose or false

   if not self.noAsserts then
      assert(tonumber(length) > 0)
      assert(tonumber(inSize) > 0)
      assert(tonumber(outSize) > 0)
   end

   self.length = tonumber(length)
   self.batchSize = opt.batchSize == nil and 1 or opt.batchSize
   self.verbose = opt.verbose or false
   self.allOutputs = opt.allOutputs or true

   self.clones = {}

   self.unitModel = UnitModel(inSize, outSize, opt)

   self.clones[1] = self.unitModel:buildUnit()
   self.params, self.gradParams = self.clones[1]:getParameters()

   local params, gradParams = self.clones[1]:parameters()
   for j = 2, self.length do
      self.clones[j] =
         self.unitModel:buildUnit(params, gradParams)
   end

   if self.verbose then
      self:message("Initialized Sequencer with " .. self.length .. " clones.")
   end

   self:reset()

   self.zeroState = torch.zeros(self.batchSize, outSize)
   self.zeroOutput = torch.zeros(self.batchSize, outSize)
   self.zeroGradState = torch.zeros(self.batchSize, outSize)

end

function Sequencer:getParameters()
   return self.params, self.gradParams
end

function Sequencer:parameters()
   return self.clones[1]:parameters()
end

function Sequencer:zeroGradParameters()
   self.clones[1]:zeroGradParameters()
end

function Sequencer:reset()
   self.crtIdx = 1
   self.crtLength = 0
end

function Sequencer:forward(input, isTraining)
   isTraining = isTraining == nil and true or isTraining
   assert(self.crtIdx <= self.length or isTraining == false)

   if isTraining then
      if self.crtIdx == 1 then
         self.clones[self.crtIdx]:forward(
            {self.zeroState, self.zeroOutput, input}
         )
      else
         self.clones[self.crtIdx]:forward({
               self.clones[self.crtIdx-1].output[1],
               self.clones[self.crtIdx-1].output[2],
               input
         })
      end
      self.crtLength = self.crtLength + 1
      self.crtIdx = self.crtIdx + 1
      return self.clones[self.crtIdx-1].output[2]
   else
      if self.crtLength == 0 then
         self.clones[self.crtIdx]:forward(
            {self.zeroState, self.zeroOutput, input}
         )
      else
         self.clones[1]:forward({
               self.clones[1].output[1]:clone(),
               self.clones[1].output[2]:clone(),
               input
         })
      end
      self.crtLength = self.crtLength + 1
      return self.clones[1].output[2]
   end
end

function Sequencer:forwardSequence(inputs, length, isTraining)
   length = length or inputs:size(1)
   isTraining = isTraining == nil and true or isTraining

   self:reset()

   local y = {}
   for i = 1, length do
      if (not isTraining) and self.allOutputs then
         y[i] = self:forward(inputs[i], isTraining):clone()
      else
         y[i] = self:forward(inputs[i], isTraining)
      end
   end

   if isTraining then
      assert((self.crtIdx == length + 1) and (self.crtLength == length))
   else
      assert((self.crtIdx == 1) and (self.crtLength == length))
   end

   if self.allOutputs then
      return y
   else
      return y[length]
   end
end

function Sequencer:backward(input, gradOutput)
   self.crtIdx = self.crtIdx - 1                    -- go back to the last clone
   local outGrad, inSignal
   if self.crtIdx == self.crtLength then
      outGrad = {self.zeroGradState, gradOutput}
   else
      if gradOutput then
         outGrad = {
            self.clones[self.crtIdx+1].gradInput[1],
            self.clones[self.crtIdx+1].gradInput[2] + gradOutput
         }
      else
         outGrad = {
            self.clones[self.crtIdx+1].gradInput[1],
            self.clones[self.crtIdx+1].gradInput[2]
         }
      end
   end
   if self.crtIdx == 1 then
      inSignal = {self.zeroState, zeroOutput, input}
   else
      inSignal = {
         self.clones[self.crtIdx-1].output[1],
         self.clones[self.crtIdx-1].output[2],
         input
      }
   end
   self.clones[self.crtIdx]:backward(inSignal, outGrad)
   return self.clones[self.crtIdx].gradInput[3]
end

function Sequencer:backwardSequence(inputs, gradOutputs)
   assert(inputs:size(1) >= self.crtLength)
   local dx = {}
   if type(gradOutputs) == "table" then
      for i = self.crtLength, 1, -1 do
         dx[i] = self:backward(inputs[i], gradOutputs[i])
      end
   else
      dx[self.crtLength] = self:backward(inputs[self.crtLength], gradOutputs)
      for i = self.crtLength-1, 1, -1 do
         dx[i] = self:backward(inputs[i])
      end
   end
   assert(self:backpropOver())
   return dx
end

function Sequencer:backpropOver()
   return self.crtIdx == 1
end


function Sequencer:message(message)
   if self.verbose then
      print(
         string.format("[LSTM Seq] "):color("blue") ..
            string.format(message):color("green")
      )
   end
end

function Sequencer:cuda()
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

   self.zeroState = self.zeroState:cuda()
   self.zeroOutput = self.zeroOutput:cuda()
   self.zeroGradState = self.zeroGradState:cuda()
end

function Sequencer:showParameters(zoom)
   self.unitModel:showParameters(self.clones[1], zoom)
end
