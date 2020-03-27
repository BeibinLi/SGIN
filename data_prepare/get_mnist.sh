#!/usr/bin/env bash

if [ -d data ]; then
    echo "data directory already present, exiting"
    exit 1
fi

mkdir mnist
wget --recursive --level=1 --cut-dirs=3 --no-host-directories --directory-prefix=mnist --accept '*.gz' http://yann.lecun.com/exdb/mnist/
pushd mnist
gunzip *
popd
