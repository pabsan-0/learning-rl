from line_profiler import LineProfiler
import torch
lp = LineProfiler()


@lp
def foo(bar):
    return bar.to(device)


device = 1
dummy = torch.Tensor([1.0]).to(device)
dict = {}

for i in range(10):
    dict[i] = foo(torch.Tensor([i]).float())

print(dict)
lp.print_stats()



def my_line_profiler(function):
    def wrapper()

        function()
        lp.print_stats()
    return wrapper


def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")
    return wrapper
