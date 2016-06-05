--------------------------------------------------------------------------------
-- This class implements the Indexing task.
-- See README.md for details.
--------------------------------------------------------------------------------

require("tasks.task")

local Indexing, Parent = torch.class("Indexing", "Task")

function Indexing:__init(opt)

   opt = opt or {}

   self.name = "Indexing"

   Parent.__init(self, opt)

   self.vectorSize = opt.vectorSize or 10
   self.mean = opt.mean or 0.5

   self.inputsInfo = {{["size"] = 1}, {["size"] = self.vectorSize}}
   self.outputsInfo = {{["size"] = self.vectorSize, ["type"] = "binary"}}

   self.targetAtTheEnd = true

   self:__initTensors()
   self:__initCriterions()

end

function Indexing:__generateBatch(Xs, Ts, Fs, L, isTraining)

   isTraining = isTraining == nil and true or isTraining

   local seqLength
   if isTraining then
      seqLength = self.trainMaxLength
   else
      seqLength = self.testMaxLength
   end

   local bs = self.batchSize

   local N = Xs[1]
   local X = Xs[2]
   local T = Ts[1]
   local F = Fs[1]

   local s = self.vectorSize

   if not self.noAsserts then
      assert(X:nDimension() == 3)                          -- check that X is 3D
      assert(X:size(1) == seqLength and X:size(2) == bs and X:size(3) == s)
      assert(N:nDimension() == 3)
      assert(N:size(1) == seqLength and N:size(2) == bs and N:size(3) == 1)
      assert(T:nDimension() == 3)
      assert(T:size(1) == 1 and T:size(2) == bs and T:size(3) == s)
      assert(F == nil)
      assert(self.fixedLength or L:nDimension() == 1 and L:size(1) == bs)
   end

   N:fill(0)

   local mean = self.mean
   X:apply(function()
         if torch.bernoulli(mean) > 0.5 then
            return self.positive
         else
            return self.negative
         end
   end)

   if not self.fixedLength then
      seqLength = torch.random(1, seqLength)
      L:fill(seqLength)
   end

   for i = 1, bs do
      n = torch.random(1, seqLength)
      T[{{1},{i},{}}]:copy(X[{{n}, {i}, {}}])
      N[1][i][1] = n
   end
end
