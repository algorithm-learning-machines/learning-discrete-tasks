--------------------------------------------------------------------------------
-- Load needed modules
--------------------------------------------------------------------------------

require("nn")
require("nngraph")
local class = require("class")

--------------------------------------------------------------------------------
-- LSTMSeq
--------------------------------------------------------------------------------

local LSTMSeq = class("LSTMSeq")

function lstm(stateSize, extSize, opt, params, gradParams)
   local inputSize = 2 * stateSize + extSize

   local fGate = opt.gateTransfer or nn.Sigmoid
   local fInput = opt.inputTransfer or nn.Tanh
   local fState = opt.stateTransfer or nn.Tanh
   local fOutput = opt.outputTransfer or nn.Tanh

   local prevState = nn.Identity()()
   local prevOutput = nn.Identity()()
   local extInput = nn.Identity()()

   local allInputs = nn.JoinTable(1)({prevState, prevOutput, extInput})
   local allLinear = nn.Linear(inputSize, 3 * stateSize)(allInputs)
   local twoGates = fGate()(nn.Narrow(1, 1, 2 * stateSize)(allLinear))
   local forgetGate = nn.Narrow(1, 1, stateSize)(twoGates)

   local writeGate = nn.Narrow(1, stateSize + 1, stateSize)(twoGates)

   local input = fInput()(nn.Narrow(1, 2*stateSize+1, stateSize)(allLinear))
   local state = nn.CAddTable()({
         nn.CMulTable()({forgetGate, prevState}),
         nn.CMulTable()({writeGate, input})
   })
   local outInputs = nn.JoinTable(1)({state, prevOutput, extInput})
   local outGate = nn.Sigmoid()(nn.Linear(inputSize, stateSize)(outInputs))
   local output = fOutput()(nn.CMulTable()({outGate, fState()(state)}))

   local lstm = nn.gModule({prevState, prevOutput, extInput}, {state, output})

   if params and gradParams then
      local crtParams, crtGradParams = lstm:parameters()
      for i = 1, #params do
         crtParams[i]:set(params[i])
         crtGradParams[i]:set(gradParams[i])
      end
   end

   return lstm
end

function LSTMSeq:__init(length, stateSize, extSize, opt)
   self.length = length
   self.stateSize = stateSize or 1
   self.extSize = extSize or 1
   self.clones = {}
   self.clones[1] = lstm(stateSize, extSize, opt)
   self.params, self.gradParams = self.clones[1]:parameters()
   for j = 2, length do
      self.clones[j] =
         lstm(stateSize, extSize, opt, self.params, self.gradParams)
   end
   self:reset()
   self.zeroState = torch.zeros(self.stateSize)
   self.zeroOutput = torch.zeros(self.stateSize)
   self.zeroGradState = torch.zeros(self.stateSize)
   self.params, self.gradParams = self.clones[1]:getParameters()
end

function LSTMSeq:reset()
   self.crtIdx = 1
   self.crtLength = 0
end

function LSTMSeq:forward(input, isTraining)
   if isTraining == nil then isTraining = true end
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
      if self.crtIdx == 1 then
         self.clones[self.crtIdx]:forward(
            {self.zeroState, self.zeroOutput, input}
         )
         self.crtIdx = self.crtIdx + 1
      else
         self.clones[self.crtIdx]:forward({
               self.clones[1].output[1]:clone(),
               self.clones[1].output[2]:clone(),
               input
         })
      end
      return self.clones[1].output[2]
   end
end

function LSTMSeq:backward(input, gradOutput)
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

function LSTMSeq:parameters()
   return self.params, self.gradParams
end

function LSTMSeq:backpropOver()
   return self.crtIdx == 1
end

return LSTMSeq
