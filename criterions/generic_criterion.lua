--------------------------------------------------------------------------------
-- Generic criterion
--
-- It receives a Task and builds a ParallelCriterion for all outputs.
-- In both forward and backward phases puts output in a table if it
-- comes as a tensor (this happens because `nngraph` doesn't wrap
-- single outputs in tables).
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
-- Import needed modules
--------------------------------------------------------------------------------

require("torch")
require("nn")
require("string.color")

local GenericCriterion = torch.class("GenericCriterion")

function GenericCriterion:__init(task, opt)

   opt = opt or {}

   self.verbose = opt.verbose or false
   self.noAsserts = opt.noAsserts or false

   local outputsInfo = task:getOutputsInfo()

   self.criterion = nn.ParallelCriterion()

   for k, v in pairs(outputsInfo) do
      if v.type == "regression" then
         self.criterion:add(nn.MSECriterion())
         self:message("MSE Criterion added")
      elseif v.type == "one-hot" then
         self.criterion:add(nn.ClassNLLCriterion())
         self:message("Class NLL Criterion added")
      elseif v.type == "binary" then
         self.criterion:add(nn.BCECriterion())
         self:message("BCE Criterion added")
      else
         assert(false, "Unknown output type")
      end
   end

   self:message("Created generic criterion")
end

function GenericCriterion:forward(Y, T)
   if type(Y) ~= "table" then
      if not self.noAsserts then assert(#T == 1) end
      return self.criterion:forward({Y}, T)
   else
      return self.criterion:forward(Y, T)
   end
end

function GenericCriterion:backward(Y, T)
   if type(Y) ~= "table" then
      if not self.noAsserts then assert(#T == 1) end
      local dY = self.criterion:backward({Y}, T)
      return dY[1]
   else
      return self.criterion:backward(Y, T)
   end
end

function GenericCriterion:cuda()
   self.criterion:cuda()
end

--------------------------------------------------------------------------------
-- Function to be used when being verbose
--------------------------------------------------------------------------------

function GenericCriterion:message(m)
   if self.verbose then
      print(string.format("[CRITERION] "):color("green") ..
               string.format(m):color("blue"))
   end
end
