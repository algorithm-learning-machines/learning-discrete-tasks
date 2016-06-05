--------------------------------------------------------------------------------
-- This class implements the Doom Clock task.
-- See README.md for details.
--------------------------------------------------------------------------------

require("tasks.task")

local DoomClock, Parent = torch.class("DoomClock", "Task")

function DoomClock:__init(opt)

   opt = opt or {}

   self.name = "Doom Clock"

   Parent.__init(self, opt)

   self.maxValue = opt.maxValue or 3
   self.mean = opt.mean or 0.3

   self.inputsInfo = {{["size"] = 2}}
   self.outputsInfo = {{["size"] = self.maxValue, ["type"] = "one-hot"}}

   self.targetAtEachStep = true

   self:__initTensors()
   self:__initCriterions()
end

function DoomClock:__generateBatch(Xs, Ts, Fs, L, isTraining)

   isTraining = isTraining == nil and true or isTraining

   local seqLength
   if isTraining then
      seqLength = self.trainMaxLength
   else
      seqLength = self.testMaxLength
   end

   local bs = self.batchSize

   local X = Xs[1]
   local T = Ts[1]
   local F = Fs[1]


   if not self.noAsserts then
      assert(X:nDimension() == 3)
      assert(X:size(1) == seqLength and X:size(2) == bs and X:size(3) == 2)
      assert(T:nDimension() == 2)
      assert(T:size(1) == seqLength and T:size(2) == bs)
      assert(F == nil)
      assert(self.fixedLength or L:nDimension() == 1 and L:size(1) == bs)
   end

   X:fill(self.negative)

   if not self.fixedLength then
      seqLength = torch.random(1, seqLength)
      L:fill(seqLength)
   end

   local mean = self.mean
   local maxValue = self.maxValue

   for i = 1, bs do
      local s = 1
      for j = 1, seqLength do
         if torch.bernoulli(mean) < 0.5 then
            X[j][i][1] = self.positive
         else
            X[j][i][2] = self.positive
            s = s + 1
            if s > maxValue then s = 1 end
         end
         T[j][i] = s
      end
   end

end
