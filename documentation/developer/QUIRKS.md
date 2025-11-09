# Quirks, Known Bugs, and Workarounds

This document collects notable or curious aspects of the codebase that stood out during development

---

## Visual Studio 2022 + pybind11 `std::mutex` bug 
(2025-11-09)

I can't believe how much time I spent on this. 
All the C++ examples worked perfectly; however, the Python test failed in `getNextAvailableInputBuffer()`, at the line:

`std::unique_lock<std::mutex> lock(this->impl->freeQueueMutex);`

Of course I assumed the error was in my own code. I tried every imaginable change for testing and debugging.  
But this time, the problem was not in my code. It turned out that in the version of the STL shipped with the Visual Studio 2022 compiler, the implementation of `std::mutex` was modified to include a `constexpr` constructor. This change causes issues when used with `pybind11` in certain situations.

The workaround is to define `_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR` in your `CMakeLists.txt` to restore the previous behavior.

More info:
- https://github.com/microsoft/STL/issues/4875
- https://stackoverflow.com/questions/78598141/first-stdmutexlock-crashes-in-application-built-with-latest-visual-studio


