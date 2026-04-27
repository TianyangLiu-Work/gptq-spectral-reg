#!/bin/bash
export PYTHONPATH="/home/tyliu/quantization:/home/tyliu/quantization2:$PYTHONPATH"
exec /home/tyliu/quantization/.conda/bin/python "$@"
