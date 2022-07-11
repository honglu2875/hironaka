#!/bin/bash

PROJECT_NAME="hironaka"
BUILD_PATH=build
CLIB_PATH=$PROJECT_NAME"/cpp"

mkdir $BUILD_PATH
g++ -fPIC -shared -o $BUILD_PATH/cppUtil.so $CLIB_PATH/cppUtil.cpp

