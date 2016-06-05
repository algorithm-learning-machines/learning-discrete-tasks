--------------------------------------------------------------------------------
-- This class implements the Copy First task.
-- See README.md for details.
--------------------------------------------------------------------------------

require("tasks.task")

local CopyFirst, Parent = torch.class("CopyFirst", "Task")

function CopyFirst:__init(opt)

   opt = opt or {}

   self.name = "Copy First"

   Parent.__init(self, opt)

   self.vectorSize = opt.vectorSize or 10
   self.mean = opt.mean or 0.5

   self.inputsInfo = {{["size"] = self.vectorSize}}
   self.outputsInfo = {{["size"] = self.vectorSize, ["type"] = "binary"}}

   self.targetAtEachStep = true

   self:__initTensors()
   self:__initCriterions()

end

function CopyFirst:__generateBatch(Xs, Ts, Fs, L, isTraining)

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
      local vsize = self.vectorSize
      assert(#Xs == 1 and #Ts == 1)
      assert(X:nDimension() == 3)
      assert(X:size(1) == seqLength and X:size(2) == bs and X:size(3) == vsize)
      assert(T:nDimension() == 3)
      assert(T:size(1) == seqLength and self.targetAtEachStep)
      assert(T:size(2) == bs and T:size(3) == vsize)
      assert(F == nil)
      assert(self.fixedLength or L:nDimension() == 1 and L:size(1) == bs)
   end

   if not self.fixedLength then
      seqLength = torch.random(1, seqLength)
      L:fill(seqLength)
   end

   local gen = function()
      if torch.bernoulli(self.mean) > 0.5 then
         return self.positive
      else
         return self.negative
      end
   end

   X:apply(gen)
   for i = 1, seqLength do
      T[i]:copy(X[1])
   end
end
