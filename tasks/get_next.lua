--------------------------------------------------------------------------------
-- This class implements the GetNext task.
-- See README.md for details.
--------------------------------------------------------------------------------

require("tasks.task")

local GetNext, Parent = torch.class("GetNext", "Task")

function GetNext:__init(opt)

   opt = opt or {}

   self.name = "Get Next"

   Parent.__init(self, opt)

   self.vectorSize = opt.vectorSize or 10
   self.inputsInfo = {
      {["size"] = self.vectorSize},
      {["size"] = self.vectorSize}
   }
   self.outputsInfo = {{["size"] = self.vectorSize, ["type"] = "binary"}}

   self.mean = opt.mean or 0.5

   self.targetAtTheEnd = true

   self:__initTensors()
   self:__initCriterions()

end

function GetNext:__generateBatch(Xs, Ts, Fs, L, isTraining)

   isTraining = isTraining == nil and true or isTraining

   local seqLength
   if isTraining then
      seqLength = self.trainMaxLength
   else
      seqLength = self.testMaxLength
   end

   local bs = self.batchSize

   local K = Xs[1]
   local X = Xs[2]
   local T = Ts[1]
   local F = Fs[1]

   local s = self.vectorSize

   if not self.noAsserts then
      assert(K:nDimension() == 3)                          -- check that X is 3D
      assert(K:size(1) == seqLength and K:size(2) == bs and K:size(3) == s)
      assert(X:nDimension() == 3)
      assert(X:size(1) == seqLength and X:size(2) == bs and X:size(3) == s)
      assert(T:nDimension() == 3)
      assert(T:size(1) == 1 and T:size(2) == bs and T:size(3) == s)
      assert(F == nil)
      assert(self.fixedLength or L:nDimension() == 1 and L:size(1) == bs)
      assert(seqLength > 1)
   end

   K:fill(self.negative)
   local mean = self.mean
   X:apply(function()
         if torch.bernoulli(mean) > 0.5 then
            return self.positive
         else
            return self.negative
         end
   end)

   if not self.fixedLength then
      seqLength = torch.random(2, seqLength)
      L:fill(seqLength)
   end

   for i = 1, bs do
      n = torch.random(1, seqLength - 1)
      T[{{1},{i},{}}]:copy(X[{{n+1}, {i}, {}}])
      K[{{1},{i},{}}]:copy(X[{{n}, {i}, {}}])
   end
end
