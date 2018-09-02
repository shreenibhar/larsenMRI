#1/bin/bash

threshnii=$1
numModels=$2
numStreams=$3
l1=$4
l2=$5
G=$6
maxVars=$7
maxSteps=$8
if [ ${#threshnii} -eq 0 -o ${#numModels} -eq 0 -o ${#numStreams} -eq 0 -o ${#l1} -eq 0 -o ${#l2} -eq 0 -o ${#G} -eq 0 -o ${#maxVars} -eq 0 -o ${#maxSteps} -eq 0 ]
then
	echo "Input consists of threshnii numModels numStreams l1 l2 G maxVars maxSteps"
	echo "Input 0 to use defaults"
	exit
fi
bash compile.sh
bash niiprocess.sh $threshnii
files=`ls datatxt`
for file in ${files[*]}
do
	echo "Solving $file"
	path=datatxt/$file
	./bin/multivariateLarsen $path $numModels $numStreams $l1 $l2 $G $maxVars $maxSteps
	for f in *.csv; do mv $f "results/$file-$f"; done
	echo "Done."
done
