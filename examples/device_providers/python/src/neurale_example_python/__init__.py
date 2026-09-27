from neurale.devices import PythonProvider


def provider():
    return PythonProvider("neurale_example_python.device:ExampleDevice")
