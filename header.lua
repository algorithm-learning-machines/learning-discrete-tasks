local oldPrint = print
function myPrint(sender, ...)
   io.write(string.color("[" .. sender .. "] ", "green"))
   oldPrint(...)
end

print = myPrint
