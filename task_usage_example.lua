require("tasks.all_tasks")

local tasks = allTasks()

for _, name in pairs(tasks) do
   print("Testing task " .. name)
   T = getTask(name)
   t = T()
end
