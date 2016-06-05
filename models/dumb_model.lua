require("torch")
require("nn")
require("nngraph")
require("string.color")

function createDumbModel(task, opt)

   opt = opt or {}

   local verbose = opt.verbose or false
   local noAsserts = opt.noAsserts or false

   local inputsInfo = task:getInputsInfo()
   local outputsInfo = task:getOutputsInfo()
   local inSize, outSize = 0, 0

   local inputs = {}

   for k, v in pairs(inputsInfo) do
      inputs[k] = nn.Identity()()
      inSize = inSize + v.size
   end

   for k, v in pairs(outputsInfo) do
      outSize = outSize + v.size
   end

   local linearNode = nn.Linear(inSize, outSize)(nn.JoinTable(2)(inputs))

   local outputs = {}
   local start = 1
   for k, v in pairs(outputsInfo) do
      outputs[k] = nn.Narrow(2, start, v.size)(linearNode)
      start = start + v.size
   end

   return nn.gModule(inputs, outputs)
end

return createDumbModel
