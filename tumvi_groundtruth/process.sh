#! /bin/bash

for line in `ls | grep ^t`
do
	filename=`echo $line|cut -d "-" -f 2|cut -d "_" -f 1`
	mv $line $filename.txt
	# echo $filename
done
