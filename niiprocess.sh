#!/bin/bash

split() {
	OIFS=$IFS
	IFS=$2
	read -a retval <<< $1
	IFS=$OIFS
}

iterate() {
	files=`ls $1`
	for file in ${files[*]}
	do
		if [ -d "$1/$file" ]
		then
			# Recur child directories
			iterate $1/$file
		else
			split $file .
			# Check file extension and decompress
			if [ ${retval[-1]} == "gz" ]
			then 
				gzip -d $1/$file
				file=`echo ${file//.gz/}`
			fi
			path=$1/$file
			# Checks if nii already processed else process it into flattened text
			# Save text file name as concatenation of file location directory structure
			tofind=`echo ${path//\//_}`
			tofind=`echo ${tofind//.nii/.txt}`
			result=$(find datatxt -name "$tofind")
			if [ ${#result} -eq 0 ]
			then
				echo "Text file: $tofind was not found:("
				echo "Processing $path..."
				cmd="\"addpath 'cpu'; write_flatten_nii('$path', $thres); exit;\""
				matlab -r -nodisplay -nosplash -nodesktop -nojvm $cmd;
				mv flat.txt datatxt/$tofind
				echo "Done:)"
			fi
		fi
	done
}

thres=$1
if [ ${#thres} -eq 0 ]
then
	echo "Input threshold for nii>|"
	exit
fi
cur="datanii"
iterate $cur
