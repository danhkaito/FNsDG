def myFun(arg1, arg2, arg3=None):
    print("arg1:", arg1)
    print("arg2:", arg2)
    print("arg3:", arg3)
 
 
# Now we can use *args or **kwargs to
# pass arguments to this function :
 
kwargs = {"arg1": "Geeks", "arg2": "for"}
myFun(**kwargs)