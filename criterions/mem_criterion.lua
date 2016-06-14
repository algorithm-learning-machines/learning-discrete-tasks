require("torch")
require("nn")
require("string.color")

local MemCriterion = torch.class("MemCriterion")

function MemCriterion:__init(task, opt)

   opt = opt or {}

   self.verbose = opt.verbose or false
   self.noAsserts = opt.noAsserts or false
   self.memorySize = opt.memorySize
   self.numEntries = opt.numEntries

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

function MemCriterion:forward(Y, T)
   local Y_truncated = Y[{{1,self.numEntries},{}}] -- get only output from mem
   --print(Y_truncated)
   if type(Y) ~= "table" then
      if not self.noAsserts then assert(#T == 1) end
      return self.criterion:forward({Y_truncated}, T)
   else
      return self.criterion:forward(Y_truncated, T)
   end
end

function MemCriterion:backward(Y, T)
   local Y_truncated = Y[{{1,self.numEntries},{}}]
   if type(Y) ~= "table" then
      if not self.noAsserts then assert(#T == 1) end
      local dY = self.criterion:backward({Y_truncated}, T)
      return dY[1]
   else
      return self.criterion:backward(Y_truncated, T)
   end
end

function MemCriterion:cuda()
   self.criterion:cuda()
end

--------------------------------------------------------------------------------
-- Function to be used when being verbose
--------------------------------------------------------------------------------

function MemCriterion:message(m)
   if self.verbose then
      print(string.format("[CRITERION] "):color("green") ..
               string.format(m):color("blue"))
   end
end
