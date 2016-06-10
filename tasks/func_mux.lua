--------------------------------------------------------------------------------
-- This class implements the Func Mux task.
-- See README.md for details.
--------------------------------------------------------------------------------

require("tasks.task")

local FuncMux, Parent = torch.class("FuncMux", "Task")

function FuncMux:__init(opt)

   opt = opt or {}

   self.name = "Func Mux"

   Parent.__init(self, opt)

   -- Configure task specific options
   self.vectorSize = opt.vectorSize or 10
   self.funcNo = math.min(opt.funcNo or 5, 5)
   self.mean = opt.mean or 0.5

   -- Configure inputs and outputs
   self.inputsInfo = {
      {["size"] = self.vectorSize, ["once"] = true},
      {["size"] = self.vectorSize, ["once"] = true},
      {["size"] = self.funcNo}
   }
   self.outputsInfo = {{["size"] = self.vectorSize, ["type"] = "binary"}}

   -- Specify what labels are available
   self.targetAtEachStep = true

   -- Call parent's functions to prepare tensors and criterions for evaluation
   self:__initTensors()
   self:__initCriterions()
end

function FuncMux:__generateBatch(Xs, Ts, Fs, L, isTraining)

   isTraining = isTraining == nil and true or isTraining

   local seqLength
   if isTraining then
      seqLength = self.trainMaxLength
   else
      seqLength = self.testMaxLength
   end

   local bs = self.batchSize

   -- References to input and target tensors
   local V1 = Xs[1]
   local V2 = Xs[2]
   local Fun = Xs[3]
   local T = Ts[1]
   local F = Fs[1]

   local vSize = self.vectorSize
   local fNo = self.funcNo

   -- Check tensors' shape
   if not self.noAsserts then
      assert(V1:nDimension() == 3 and V1:size(1) == seqLength
                and V1:size(2) == bs and V1:size(3) == vSize)
      assert(V2:nDimension() == 3 and V2:size(1) == seqLength
                and V2:size(2) == bs and V2:size(3) == vSize)
      assert(Fun:nDimension() == 3 and Fun:size(1) == seqLength
                and Fun:size(2) == bs and Fun:size(3) == fNo)
      assert(T:nDimension() == 3 and T:size(1) == seqLength
                and T:size(2) == bs and T:size(3) == vSize)
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
   V1:zero()
   V2:zero()
   V1[1]:apply(gen)
   V2[1]:apply(gen)

   local fEq = function (val1, val2) return math.abs(val1 - val2) < 0.00001 end

   local fNeg = function (b1)
      return fEq(b1, self.positive) and self.negative or self.positive
   end

   local fOr = function (b1, b2)
      local r = fEq(b1, self.positive) or fEq(b2, self.positive)
      return r and self.positive or self.negative
   end

   local fAnd = function (b1, b2)
      local r = fEq(b1, self.positive) and fEq(b2, self.positive)
      return r and self.positive or self.negative
   end

   local fXor = function (b1, b2)
      local p1 = fEq(b1, self.positive)
      local p2 = fEq(b2, self.positive)
      local r = (p1 and not p2) or (not p1 and p2)
      return r and self.positive or self.negative
   end

   local fNand = function (b1, b2)
      local r = fEq(b1, self.negative) or fEq(b2, self.negative)
      return r and self.positive or self.negative
   end

   local funcs = {fNeg, fOr, fAnd, fXor, fNand}

   Fun:fill(self.negative)

   for i = 1, bs do
      local v1 = V1[1][i]:clone()
      local v2 = V2[1][i]:clone()

      print("v1:")
      print(v2)
      print("v2:")
      print(v2)
      print("out:")

      for t = 1, seqLength do
         local fIdx = torch.random(1, fNo)
         Fun[t][i][fIdx] = self.positive

         print("func: " .. fIdx)
         v2:map(v1, funcs[fIdx])
         T[t][i]:copy(v2)
         print(v2)
      end
      print("----")
   end

end
