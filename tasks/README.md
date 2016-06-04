# Tasks

## TL;DR

Each task should be implemented as a class that inherits `tasks.Task`.


    require("tasks.task")
    local YourTask, Parent = torch.class("YourTask", "Task")


Implement `YourTask:__init(opt)` and set `self.inputsInfo` and
`self.outputsInfo`. Set `self.targetAtEachStep` to `true` if you
provide target values at each step. Set `self.targetAtTheEnd` to
`true` if there is only one label at the end of the sequence.

Implement `YourTask:__generateBatch(Xs, Ts, Fs, L, isTraining)` where:

 - `Xs` is a table with a tensor for each input: `seqLength x
   batchSize x inputSize`

 - `Ts` is a table with a tensor for each output: `(seqLength|1) x
   batchSize x outputSize`

 - `Fs` is a table with a tensor with flags for the steps when labels
   are provided; if `self.targetAtEachStep` or `self.targetAtTheENd`
   is `true` then `Fs` is empty

 - `L` is `nil` if `self.fixedLength` is `true` or a tensor with
   length `batchSize` otherwise

## General options

The following fields can be configured through the `opt` argument:

 - `batchSize` - this is important as tensors with given size are prepared
 - `positive` and `negative` (e.g. 0/1 or -1/1)
 - `trainMaxLength` - maximum length for train sequences
 - `testMaxLength` - maximum length for test sequences
 - `fixedLength` (`true` if all train / test sequences should have the
   maximum length)
 - `onTheFly` - `true` to generate batches on the fly
 - `trainSize` and `testSize` set the number of examples to be
   generated if `onTheFly` is `false`

 - `verbose`
 - `noAsserts`

Of course, other task specific options might be configured.

## API

 - `__init(opt)` - constructor
 - `updateBatch(split)` - split should be `"train"` or `"test"`
 - `isEpochOver(split)`
 - `resetIndex(split)`
 - `getInputsInfo()`
 - `getOutputsInfo()`
 - `evaluateBatch(output, targets, err)`
 - `hasTargetAtEachStep()`
 - `hasTargetAtTheEnd()`
 - `cuda()`
 - `displayCurrentBatch()`


## inputsInfo

`self.inputsInfo` should be a Lua array (a table) with an entry for
each separate input. Each entry is a table with at least the `"size"`
key.

## outputsInfo

`self.outputsInfo` should be a Lua array (a table) with an entry for
each separate output. Each entry is a table with `"size"` and
`"type"`. The value for `"type"` should be one of the following:
`"one-hot", "regression", "binary"`.

## Implemented tasks

### Doom Clock

This tasks assumes you have an **internal state**. You receive an
one-hot encoded vector of size 2. When the second value is positive
you have to switch an internal state. The output at each step must be
the internal state.

### Get Next

This tasks goes for **key** addressing. You receive a *key* and a
sequence of vectors. At the end of the sequence you must return the
next value after the *key*.

### Indexing

You receive a number which represents the index of the desired
vector. Another input receives the vectors from a sequence one by
one. At the end of the sequence you must return the nth vector.

### Copy First

This tasks requires the model to remember the first vector from a
sequence and reproduce it at each step.

Options:

 - `vectorSize`

### Copy

Copy input to output. The model must learn the identity function.

### Binary Sum

You receive two sequences of binary signals. You must sum them.

### Substract On Signal

You are given two numbers A and B and a sequence of binary
signals. You start with D, a number you must output, equal to A. Every
time the signal is positive you substract B from D. You must also
output the number of positive signals and if D is positive.
