require("tasks.task")

local __allTasks = {
   -- ["CopyFirst"]         =  "tasks.copy_first",
   ["Copy"]              =  "tasks.copy",
   -- ["DoomClock"]         =  "tasks.doom_clock",
   -- ["Indexing"]          =  "tasks.indexing",
   -- ["GetNext"]           =  "tasks.get_next",
    -- ["BinarySum"]         =  "tasks.binary_sum",
   -- ["SubtractOnSignal"]  =  "tasks.subtract_on_signal"
}

function allTasks()
   local _allTasks = {}
   for taskName, _ in pairs(__allTasks) do table.insert(_allTasks, taskName) end
   return _allTasks
end

function getTask(taskName)
   assert(__allTasks[taskName] ~= nil, "Task not available!")

   require(__allTasks[taskName])

   -- Remember the old value
   local __oldVal = __TaskType

   -- Put task type in global variable __TaskType
   loadstring("__TaskType = " .. taskName)()

   -- Copy task type to local _TaskType
   local _TaskType = __TaskType

   -- Put back the old value of global varialbe __TaskType
   __TaskType = __oldVal

   -- Return the value of local _TaskType
   return _TaskType
end
