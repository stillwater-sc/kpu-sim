#!/usr/bin/env bash
kpu-dfg-gen --template matmul -M 1024 -N 1024 -K 1024 --tiles 4x4x4 -o dfg.json
kpu-dfg-sched -i dfg.json -o scheduled.json --algorithm ASAP
kpu-dfg-compile -i scheduled.json -o programs.json
kpu-dfg-viz -i scheduled.json -o timeline.json --format chrome-trace
kpu-dfg-analyze -i dfg.json --stats --critical-path
