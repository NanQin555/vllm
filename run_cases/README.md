# If you wanna build your own VLLM with source code, follow commands below.

Use ccache to accelerate compile speed, of course you can build VLLM without ccache
```
sudo apt install ccache
```

Create virtual environment 
Choose the python version you have (3.10 - 3.13)
```
uv venv --python 3.12 --seed
source .venv/bin/activate
```

Limit the number of compilation threads. If the flag is set to 16 
and still uses too many hardware resources, set flag to 8 or less.
```
export CMAKE_BUILD_PARALLEL_LEVEL=16
```

# build command
Use pip cache, because before compile will download and install some wheel, if you have
pip cache can skip this stage.
Only compile with ccache, can you set flag CCACHE_NOHASHDIR="true"
"-i https://pypi.tuna.tsinghua.edu.cn/simple" is not necessary.
```
export MAKEFLAGS="-j16"
export MAX_JOBS=16
export CMAKE_BUILD_PARALLEL_LEVEL=16
export PIP_CACHE_DIR=$HOME/.cache/pip· 
CCACHE_NOHASHDIR="true" python3.11 -m pip install \
    --cache-dir $PIP_CACHE_DIR \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --no-build-isolation -v -e . 
```


# Debug VLLM
If you wanna dubug vllm, we need compile vllm with Debug model
```
export MAKEFLAGS="-j24"
export MAX_JOBS=24
export CMAKE_BUILD_PARALLEL_LEVEL=24
export CMAKE_BUILD_TYPE=Debug
export PIP_CACHE_DIR=$HOME/.cache/pip
CCACHE_NOHASHDIR="true" python3.11 -m pip install \
    --cache-dir $PIP_CACHE_DIR \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --no-build-isolation -v -e . 
```
# stop tasks
```
ps -ef | grep -E "nvcc|cicc|cclplus|ptxas"
pkill -9 -f "cicc|cclplus|ptxas|nvcc"
```

# run test
```
# /path/to/your/vllm/project run command:
pytest --ignore=tests/distributed --ignore=tests/tpu --ignore=tests/v1/engine --ignore=tests/models/multimodal
pytest tests/entrypoints/llm
```