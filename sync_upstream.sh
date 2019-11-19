#!/bin/bash
# git config --global credential.helper cache
# Usage: ./sync_upstream.sh &>/dev/null
while [ 1 ]
do
	is_src_changed=$(git status | grep "src/")
	if [ ! -z "$is_src_changed" ]
	then
		git add -A src/
		git commit --amend --no-edit --allow-empty
		git push -f origin sync
	fi
	sleep 6
done
