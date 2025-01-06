# NOTE: on MacOS you need to add an addition flag: -undefined dynamic_lookup
default:
	c++ -O3 -Wall -shared -std=c++17 -fPIC \
	-I/Users/bytedance/.pyenv/versions/3.12.8/include/python3.12 -I/Users/bytedance/.pyenv/versions/3.12.8/lib/python3.12/site-packages/pybind11/include \
  src/simple_ml_ext.cpp -o src/simple_ml_ext.so \
	-lintl -ldl -L/Users/bytedance/.pyenv/versions/3.12.8/lib -Wl,-rpath,/Users/bytedance/.pyenv/versions/3.12.8/lib -L/opt/homebrew/lib -Wl,-rpath,/opt/homebrew/lib -framework CoreFoundation -lpython3.12


