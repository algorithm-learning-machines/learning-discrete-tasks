local Task = require("tasks.task")

local Copy, Parent = torch.class("Copy", "Task")

function Copy:__init(opt)
   opt = opt or {}

   Parent.__init(self, opt)

   self.isClassification = true

   self.inputSize = opt.inputSize or 10
   self.outputSize = self.inputSize

   self.targetAtEachStep = true

   self:initTensors()
end

function Copy:generateBatch(X, T, F, L, isTraining)

   isTraining = isTraining == nil and true or isTraining

   local seqLength
   if isTraining then
      seqLength = self.trainMaxLength
   else
      seqLength = self.testMaxLength
   end

   local bs = self.batchSize

   if not self.noAsserts then
      local ins = self.inputSize
      local outs = self.outputSize

      assert(X:nDimension() == 3)
      assert(X:size(1) == seqLength and X:size(2) == bs and X:size(3) == ins)
      assert(T:nDimension() == 3)
      assert((T:size(1) == seqLength or T:size(1) == 1 and self.targetAtTheEnd))
      assert(T:size(2) == bs and T:size(3) == outs)
      assert((self.targetAtEachStep or self.targetAtTheEnd) or F ~= nil)
      assert(F == nil or F:size(1) == T:size(1) and F:size(2) == T:size(2)
                and F:size(3) == T:size(3))
      assert(self.fixedLength or L:nDimension() == 1 and L:size(1) == bs)
   end

   if not self.fixedLength then
      seqLength = torch.random(1, seqLength)
      L:fill(seqLength)
   end


   local gen = function()
      if torch.bernoulli(0.6) > 0.5 then
         return self.positive
      else
         return self.negative
      end
   end

   X:apply(gen)
   T:copy(X)
end
