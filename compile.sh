#!/bin/bash

if [ ! -d "datanii" ]
then
	mkdir datanii
fi
if [ ! -d "datatxt" ]
then
	mkdir datatxt
fi
if [ ! -d "results" ]
then
	mkdir results
fi
cd bin
make
