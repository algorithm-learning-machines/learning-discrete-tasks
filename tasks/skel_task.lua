--------------------------------------------------------------------------------
-- This is a skel code one might use as a starting point when
-- implementing tasks.
--------------------------------------------------------------------------------

require("tasks.task")

local TaskName, Parent = torch.class("TaskName", "Task")

function TaskName:__init(opt)

   opt = opt or {}

   self.name = "TaskName"

   Parent.__init(self, opt)

   -- Configure task specific options
   self.vectorSize = opt.vectorSize or 10
   self.mean = opt.mean or 0.5

   -- Configure inputs and outputs
   self.inputsInfo = {{["size"] = self.vectorSize}}
   self.outputsInfo = {{["size"] = self.vectorSize, ["type"] = "binary"}}

   -- Specify what labels are available
   self.targetAtEachStep = true

   -- Call parent's functions to prepare tensors and criterions for evaluation
   self:__initTensors()
   self:__initCriterions()
end

function TaskName:__generateBatch(Xs, Ts, Fs, L, isTraining)

   isTraining = isTraining == nil and true or isTraining

   local seqLength
   if isTraining then
      seqLength = self.trainMaxLength
   else
      seqLength = self.testMaxLength
   end

   local bs = self.batchSize

   -- References to input and target tensors
   local X = Xs[1]
   local T = Ts[1]
   local F = Fs[1]

   -- Check tensors' shape
   if not self.noAsserts then
      local vSize = self.vectorSize

      assert(X:nDimension() == 3)
      assert(X:size(1) == seqLength and X:size(2) == bs and X:size(3) == vSize)
      assert(T:nDimension() == 3)
      assert(T:size(1) == seqLength and self.targetAtEachStep)
      assert(T:size(2) == bs and T:size(3) == vSize)
      assert(F == nil)
      assert(self.fixedLength or L:nDimension() == 1 and L:size(1) == bs)
   end

   -- All sequences in a batch must have the same length
   if not self.fixedLength then
      seqLength = torch.random(1, seqLength)
      L:fill(seqLength)
   end

   -- Generate data
   local gen = function()
      if torch.bernoulli(self.mean) > 0.5 then
         return self.positive
      else
         return self.negative
      end
   end

   X:apply(gen)
   T:copy(X)
end
