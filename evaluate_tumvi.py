import cv2
import numpy as np
import glob
import os.path as osp
import os

import datetime
from tqdm import tqdm

from dpvo.utils import Timer
from dpvo.dpvo import DPVO
from dpvo.stream import image_stream
from dpvo.config import cfg

import torch
from multiprocessing import Process, Queue

### evo evaluation library ###
import evo
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from evo.core import sync
import evo.main_ape as main_ape
from evo.core.metrics import PoseRelation


SKIP = 0

def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)

@torch.no_grad()
def run(cfg, network, imagedir, calib, stride=1, viz=False, fisheye=True):

    slam = None

    queue = Queue(maxsize=8)
    reader = Process(target=image_stream, args=(queue, imagedir, calib, stride, 0, fisheye))
    reader.start()

    while 1:
        (t, image, intrinsics) = queue.get()
        if t < 0: break

        image = torch.from_numpy(image).permute(2,0,1).cuda()
        intrinsics = torch.from_numpy(intrinsics).cuda()

        if viz: 
            show_image(image, 1)

        if slam is None:
            slam = DPVO(cfg, network, ht=image.shape[1], wd=image.shape[2], viz=viz)

        image = image.cuda()
        intrinsics = intrinsics.cuda()

        with Timer("SLAM", enabled=False):
            slam(t, image, intrinsics)

    for _ in range(12):
        slam.update()

    reader.join()

    return slam.terminate()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='dpvo.pth')
    parser.add_argument('--config', default="config/default.yaml")
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--tumvidir', default="datasets/tum-vi")
    parser.add_argument('--log', type=str)
    parser.add_argument('--savetraj', action="store_true")
    args = parser.parse_args()

    cfg.merge_from_file(args.config)

    print("\nRunning with config...")
    print(cfg, "\n")

    torch.manual_seed(1234)

    tumvi_scenes = [
        # "corridor1",
        # "corridor2",
        # "corridor3",
        # "corridor4",
        # "corridor5",
        # "magistrale1",
        # "magistrale2",
        # "magistrale3",
        # "magistrale4",
        # "magistrale5",
        # "magistrale6",
        # "outdoors1",
        # "outdoors2",
        # "outdoors3",
        # "outdoors4",
        # "outdoors5",
        # "outdoors6",
        # "outdoors7",
        # "outdoors8",
        "room1",
        "room2",
        "room3",
        "room4",
        "room5",
        "room6",
        # "slides1",
        # "slides2",
        # "slides3",
    ]

    results = {}
    for scene in tumvi_scenes:
        imagedir = os.path.join(args.tumvidir, scene, "mav0/cam0/data")
        groundtruth = "datasets/tumvi_groundtruth/{}.txt".format(scene) 

        scene_results = []
        for i in range(args.trials):
            traj_est, timestamps = run(cfg, args.network, imagedir, "calib/tumvi.txt", args.stride, args.viz)

            images_list = sorted(glob.glob(os.path.join(imagedir, "*.png")))[::args.stride]
            tstamps = [float(x.split('/')[-1][:-4]) for x in images_list]

            traj_est = PoseTrajectory3D(
                positions_xyz=traj_est[:,:3],
                orientations_quat_wxyz=traj_est[:,3:],
                timestamps=np.array(tstamps))


            traj_ref = file_interface.read_tum_trajectory_file(groundtruth)
            traj_est.timestamps /= 1e9
            traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
            if args.savetraj:
                file_interface.write_tum_trajectory_file('traj/tumvi/' + scene+'est.txt',traj_est)
                file_interface.write_tum_trajectory_file('traj/tumvi/' + scene+'gt.txt',traj_ref)
            result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
                pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)

            scene_results.append(result.stats["rmse"])

        results[scene] = np.median(scene_results)
        if args.log:
            with open(args.log, 'a+', encoding='utf-8') as f:
                print(scene, sorted(scene_results),file=f)
                print(scene, sorted(scene_results))
        else:
            print(scene, sorted(scene_results))
        # print(scene, sorted(scene_results))

    xs = []
    for scene in results:
        if args.log:
            with open(args.log, 'a+', encoding='utf-8') as f:
                print(scene, results[scene], end='\n', file=f)
                print(scene, results[scene])
        else:
            print(scene, results[scene])
        # print(scene, results[scene])
        xs.append(results[scene])

    if args.log:
        with open(args.log, 'a+', encoding='utf-8') as f:
            print("AVG: ", np.mean(xs), end='\n', file=f)        
            print("AVG: ", np.mean(xs))
    else:
        print("AVG: ", np.mean(xs))
    # print("AVG: ", np.mean(xs))

    

    
