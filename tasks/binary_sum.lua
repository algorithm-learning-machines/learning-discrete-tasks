--------------------------------------------------------------------------------
-- This class implements the Binary Sum task.
-- See README.md for details.
--------------------------------------------------------------------------------

require("tasks.task")

local BinarySum, Parent = torch.class("BinarySum", "Task")

function BinarySum:__init(opt)

   opt = opt or {}

   self.name = "Binary Sum"

   Parent.__init(self, opt)

   self.inputsNo = opt.inputsNo or 10
   self.mean = opt.mean or 0.5

   self.inputsInfo = {}
   for i = 1, self.inputsNo do
      self.inputsInfo[i] = {["size"] = 2}
   end
   self.outputsInfo = {{["size"] = 2, ["type"] = "one-hot"}}

   self.targetAtEachStep = true

   self:__initTensors()
   self:__initCriterions()

end

function BinarySum:__generateBatch(Xs, Ts, Fs, L, isTraining)

   isTraining = isTraining == nil and true or isTraining

   local seqLength
   if isTraining then
      seqLength = self.trainMaxLength
   else
      seqLength = self.testMaxLength
   end

   local bs = self.batchSize

   local T = Ts[1]
   local F = Fs[1]

   if not self.noAsserts then
      assert(#Xs == self.inputsNo)
      for i = 1, #Xs do
         local a = Xs[i]:size()
         assert(a[1] == seqLength and a[2] == bs and a[3] == 2)
      end
      assert(T:size(1) == seqLength and T:size(2) == bs)
      assert(F == nil)
      assert(self.fixedLength or L ~= nil and L:size(1) == bs)
   end

   if not self.fixedLength then
      seqLength = torch.random(1, seqLength)
      L:fill(seqLength)
   end

   for i = 1, #Xs do
      Xs[i]:fill(self.negative)
   end
   T:fill(1)

   local mean = self.mean

   for i = 1, bs do
      local carry = 0
      for t = 1, seqLength do
         for j = 1, #Xs do
            local z = torch.bernoulli(mean) + 1
            Xs[i][{t, i, z}] = self.positive
            if z == 2 then carry = carry + 1 end
         end
         if carry % 2 == 1 then
            T[{t, i}] = 2
            carry = carry - 1
         end
         carry = carry / 2
      end
   end

end
