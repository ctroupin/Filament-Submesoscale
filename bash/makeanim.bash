#!/bin/bash

ffmpeg -framerate 12 -pattern_type glob -i "SSTanomalies_*.jpg" -c:v libx264 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" SST_anomalies_global_20240709.mp4
