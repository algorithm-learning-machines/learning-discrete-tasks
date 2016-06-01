require("torch")

require("models.lstm")
require("models.sequencer")

local LSTMClassifier = torch.class("LSTMClassifier")

function LSTMClassifier:__init(task, opt)
   opt = opt or {}

   self.noAsserts = opt.noAsserts or false
   self.verbose = opt.verbos or false


   if task:hasTargetAtTheEnd() then
      self.allOutputs = false
   else
      self.allOutputs = true
   end

   self.inSize = task:getInputSize()
   self.outSize = task:getOutputSize()
   self.targetsNo = task.getOutputsNo()
   self.batchSize = task.batchSize -- NOT NICE!

   assert(task:getType() == "classification")

   local inSize, outSize = self.inSize, self.outSize

   self.model = Sequencer(task.trainMaxLength, inSize, outSize, LSTM, opt)
   self.lastLayer = nn.SoftMax()
end

function LSTMClassifier:getParameters()
   return self.model:getParameters()
end

function LSTMClassifier:parameters()
   return self.model:parameters()
end

function LSTMClassifier:zeroGradParameters()
   return self.model:zeroGradParameters()
end


function LSTMClassifier:forward(X, l, isTraining)
   if not self.noAsserts then
      assert(
         (X:nDimension() == 3)
            and (X:size(1) >= l)
            and (X:size(3) == self.inSize)
      )
   end

   self.O = self.model:forwardSequence(X, l, isTraining)

   if self.allOutputs then
      if not self.noAsserts then
         assert(#self.O == l)
      end
      self.output = {}
      for i = 1, l do
         if not self.noAsserts then
            assert(self.O[i]:nDimension() == 2)
            assert(self.O[i]:size(2) == self.targetsNo * self.outSize)
         end

         self.output[i] = self.lastLayer:forward(
            self.O[i]:view(self.batchSize * self.targetsNo, self.outSize)
         )
      end
   else
      if not self.noAsserts then
         assert(self.O:nDimension() == 2)
         assert(self.O:size(2) == self.targetsNo * self.outSize)
      end

      self.output = self.lastLayer:forward(
         self.O:view(self.batchSize * self.targetsNo, self.outSize)
      )
   end

   return self.output
end

function LSTMClassifier:backward(X, dOutputs, l)
   if not self.noAsserts then
      assert(
         (X:nDimension() == 3)
            and (X:size(1) >= l)
            and (X:size(2) == self.batchSize)
            and (X:size(3) == self.inSize)
      )
   end

   local dO
   if self.allOutputs then
      if not self.noAsserts then
         assert(#dOutputs == l)
      end

      dO = {}
      for i = 1, l do
         if not self.noAsserts then
            assert(dOutputs[i]:nDimension() == 2)
            assert(dOutputs[i]:size(1) == self.batchSize * self.targetsNo)
            assert(dOutputs[i]:size(2) == self.outSize)
         end

         dO[i] = self.lastLayer:backward(
            self.O[i]:view(self.batchSize * self.targetsNo, self.outSize),
            dOutputs[i]:view(self.batchSize * self.targetsNo, self.outSize)
         )
      end

      self.model:backwardSequence(X, dO)
   else
      if not self.noAsserts then
         assert(dOutputs:nDimension() == 2)
         assert(dOutputs:size(1) == self.batchSize * self.targetsNo)
         assert(dOutputs:size(2) == self.outSize)
      end

      dO = self.lastLayer:backward(
         self.O:view(self.batchSize * self.targetsNo, self.outSize),
         dOutputs:view(self.batchSize * self.targetsNo, self.outSize)
      )

      self.model:backwardSequence(X, dO)
   end
end

function LSTMClassifier:cuda()
   require("cunn")
   self.model:cuda()
   self.lastLayer:cuda()
end
