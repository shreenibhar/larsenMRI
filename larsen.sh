#!/bin/bash

numModels=$1
numStreams=$2
l1=$3
l2=$4
G=$5
maxVars=$6
maxSteps=$7
if [ ${#numModels} -eq 0 -o ${#numStreams} -eq 0 -o ${#l1} -eq 0 -o ${#l2} -eq 0 -o ${#G} -eq 0 -o ${#maxVars} -eq 0 -o ${#maxSteps} -eq 0 ]
then
	echo "Input consists of numModels numStreams l1 l2 G maxVars maxSteps"
	echo "Input 0 to use defaults"
	exit
fi
files=`ls datatxt`
for file in ${files[*]}
do
	echo "Solving $file"
	path=datatxt/$file
	./bin/multivariateLarsen $path $numModels $numStreams $l1 $l2 $G $maxVars $maxSteps
	for f in *.csv; do mv $f "results/$file-$f"; done
	echo "Done."
done
