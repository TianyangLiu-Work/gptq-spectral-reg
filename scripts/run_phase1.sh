#!/bin/bash
export PYTHONPATH="$PWD:/home/tyliu/quantization:$PYTHONPATH"
exec /home/tyliu/quantization/.conda/bin/python "$@"
