#!/bin/bash 
for C in 5 7 11 
do 
	wget -c http://iie.fing.edu.uy/~nacho/data/energia/darset/c$C.7z
	7zr -x c$C.7z
done
