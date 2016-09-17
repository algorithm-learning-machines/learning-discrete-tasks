require("nn")
require("rnn")
require("nngraph")
require("string.color")

--------------------------------------------------------------------------------
-- Creates a wrapper for rnn.LSTM .
--------------------------------------------------------------------------------

local LSTMWrapper = torch.class("LSTMWrapper")

function LSTMWrapper:__init(outSize, inputSize, opt)

   opt = opt or {}

   self.noAsserts = opt.noAsserts or false
   self.verbose = opt.verbose or false

   -- Use --noAsserts not to check size
   if not self.noAsserts then
      --assert(tonumber(stateSize) > 0)
      --assert(tonumber(extSize) > 0)
   end

   -- Should this LSTM cell be prepared to process batches
   local batchSize = opt.batchSize == nil and 1 or opt.batchSize
   if batchSize >= 1 then
      self.fsDim = 2
   else
      self.fsDim = 1
   end

   self.inputSize = inputSize 
   self.outputSize = outputSize 
   self.rho = opt.rho or 5 -- backprop steps

   self.model = nn.LSTM(self.inputSize, self.outputSize, self.rho)

end

--------------------------------------------------------------------------------
-- Basic wrapper functions over LSTM model
--------------------------------------------------------------------------------

function LSTMWrapper:forward(input, gradOutput)
   return self.model:forward(input, gradOuput)
end


function LSTMWrapper:backward(input, gradOutput)
   return self.model:backward(input, gradOutput)
end


function LSTMWrapper:updateOutput(input)
   return self.model:updateOutput(input)
end

function LSTMWrapper:updateGradInput(input, gradOutput)
   return self.model:updateGradInput(input, gradOutput)
end

