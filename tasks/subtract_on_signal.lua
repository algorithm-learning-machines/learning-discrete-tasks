--------------------------------------------------------------------------------
-- This class implements the Subtract On Signal task.
-- See README.md for details.
--------------------------------------------------------------------------------

require("tasks.task")

local SubtractOnSignal, Parent = torch.class("SubtractOnSignal", "Task")

function SubtractOnSignal:__init(opt)

   opt = opt or {}

   self.name = "SOS"

   Parent.__init(self, opt)

   self.maxValue = opt.maxValue or 20
   self.mean = opt.mean or 0.3

   self.inputsInfo = {{["size"] = 1}, {["size"] = 1}, {["size"] = 2}}
   self.outputsInfo = {
      {["size"] = 1, ["type"] = "regression"},                         -- result
      {["size"] = 1, ["type"] = "regression"},               -- how many signals
      {["size"] = 2, ["type"] = "one-hot"}                     -- still positive
   }

   self.targetAtEachStep = true

   self:__initTensors()
   self:__initCriterions()

end

function SubtractOnSignal:__generateBatch(Xs, Ts, Fs, L, isTraining)

   isTraining = isTraining == nil and true or isTraining

   local seqLength
   if isTraining then
      seqLength = self.trainMaxLength
   else
      seqLength = self.testMaxLength
   end

   local bs = self.batchSize

   local A = Xs[1]
   local B = Xs[2]
   local S = Xs[3]
   local D = Ts[1]
   local N = Ts[2]
   local Z = Ts[3]

   if not self.noAsserts then
      assert(#Xs == 3 and #Ts == 3)
      assert(A:size(1) == seqLength and A:size(2) == bs and A:size(3) == 1)
      assert(B:size(1) == seqLength and B:size(2) == bs and B:size(3) == 1)
      assert(S:size(1) == seqLength and S:size(2) == bs and S:size(3) == 2)
      assert(D:size(1) == seqLength and D:size(2) == bs and D:size(3) == 1)
      assert(N:size(1) == seqLength and N:size(2) == bs and N:size(3) == 1)
      assert(Z:size(1) == seqLength and Z:size(2) == bs)
      assert(#Fs == 0)
      assert(self.fixedLength or L:nDimension() == 1 and L:size(1) == bs)
   end

   if not self.fixedLength then
      seqLength = torch.random(1, seqLength)
      L:fill(seqLength)
   end

   A:fill(0)
   B:fill(0)
   Z:fill(self.negative)

   A[1]:apply(function() return torch.random(2, self.maxValue) end)
   B[1]:map(A[1], function(b, a) return torch.random(1, a/2 + 1) end)

   local mean = self.mean

   for i = 1, bs do
      local a, b, s, d, n, z
      a = A[1][i][1]
      b = B[1][i][1]
      d = a
      n = 0
      z = 1
      for t = 1, seqLength do
         s = torch.bernoulli(mean)
         S[{t,i,s+1}] = self.positive
         if s > 0.5 then
            d = d - b
            n = n + 1
            if d <= 0 then z = 2 end
         end
         D[{t,i,1}] = d
         N[{t,i,1}] = n
         Z[{t,i}] = z
      end
   end
end
