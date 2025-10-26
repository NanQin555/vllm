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
export PIP_CACHE_DIR=$HOME/.cache/pip
CCACHE_NOHASHDIR="true" python3.11 -m pip install \
    --no-build-isolation -v -e . \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --cache-dir $PIP_CACHE_DIR
```