#! /bin/bash
set -e

for line in `ls ../datasets/kitti_groundtruth`
do
    seq=`echo $line|cut -d "." -f 1`
    python3 kitti_poses_and_timestamps_to_trajectory.py ../datasets/kitti_groundtruth/${line} ../datasets/kitti_odometry/${seq}/times.txt ../datasets/kitti_groundtruth/tum_${seq}.txt
done