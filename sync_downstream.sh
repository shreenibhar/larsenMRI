#!/bin/bash
# Usage: ./sync_downstream.sh &>/dev/null
while [ 1 ]
do
	git fetch origin sync
	git reset --hard origin/sync
	sleep 6
done
