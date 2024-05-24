#!/bin/bash
set -e -u -x


function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$wheel" -w /io/wheelhouse_repaired/
    fi
}

# Install a system package required by our library
# yum install -y atlas-devel
# yum install -y eigen3-devel
# yum install -y boost-devel
# yum install -y wget

# Compile wheels

# for PYBIN in /opt/python/*/bin; do
# PYBIN=/opt/python/cp38-cp38/bin

variable=(
# /opt/python/cp36-cp36/bin
# /opt/python/cp37-cp37/bin
/opt/python/cp38-cp38/bin
/opt/python/cp39-cp39/bin
/opt/python/cp310-cp310/bin
/opt/python/cp311-cp311/bin
/opt/python/cp312-cp312/bin
)

for PYBIN in "${variable[@]}"; do
  # "${PYBIN}/pip" install -r /io/dev-requirements.txt
  "${PYBIN}/pip" wheel /io/ --no-deps -w wheelhouse_raw/
done








# done

# Bundle external shared libraries into the wheels
for whl in wheelhouse_raw/*.whl; do
    repair_wheel "$whl"
done

# Install packages and test
# for PYBIN in /opt/python/*/bin/; do
# PYBIN=/opt/python/cp38-cp38/bin
# "${PYBIN}/pip" install python-manylinux-demo --no-index -f /io/wheelhouse
# (cd "$HOME"; "${PYBIN}/nosetests" pymanylinuxdemo)
# done
