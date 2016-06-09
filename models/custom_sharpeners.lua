--------------------------------------------------------------------------------
-- Custom sharpeners used in the model
--------------------------------------------------------------------------------

require "nn"

--------------------------------------------------------------------------------
-- Simple sharpener that amplifies small differences by multiplying 
-- with a large constant before softmaxing
--------------------------------------------------------------------------------
local mul_soft_max, Parent = torch.class("nn.mul_soft_max", "nn.Module")

function mul_soft_max:__init(constant)
   Parent.__init(self)
   self.constant = constant or 300
   seq = nn.Sequential()
   seq:add(nn.MulConstant(self.constant, true))
   seq:add(nn.SoftMax())
   self.seq = seq
end

function mul_soft_max:updateOutput(input)
   self.output = self.seq:updateOutput(input)
   return self.output
end

function mul_soft_max:updateGradInput(input, gradOutput)
   self.gradInput = self.seq:updateGradInput(input, gradOutput)
   return self.gradInput
end


return mul_soft_max
