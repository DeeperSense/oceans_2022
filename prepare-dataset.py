import cv2
import numpy as np
import pathlib
import argparse
from tqdm.contrib import tzip

parser = argparse.ArgumentParser()
parser.add_argument(
    "--sonar_dir",
    type=pathlib.Path,
    required=True,
    help="input directory for sonar images",
)
parser.add_argument(
    "--camera_dir",
    type=pathlib.Path,
    required=True,
    help="input directory for camera images",
)
parser.add_argument(
    "--output_dir",
    type=pathlib.Path,
    required=True,
    help="output directory for dataset",
)
opt = parser.parse_args()

darkness_factors = [0, 0.20, 0.50, 0.75, 0.90, 0.95, 1]

camera_in_dir = opt.camera_dir
sonar_in_dir = opt.sonar_dir

camera_in_list = sorted(camera_in_dir.glob("**/*.png"))
sonar_in_list = sorted(sonar_in_dir.glob("**/*.png"))

for d in darkness_factors:
    dataset_out_dir = opt.output_dir / "dark{}".format(int(d * 100))
    pathlib.Path.mkdir(dataset_out_dir / "train", parents=True, exist_ok=True)
    pathlib.Path.mkdir(dataset_out_dir / "test", parents=True, exist_ok=True)
    var = 100 * d
    k = int(np.ceil(100 * d) // 2 * 2 + 1)
    kernel = (k, k)  # ceil to immediate odd value

    print("Creating dataset for darkness level: {} at {}".format(d, dataset_out_dir))
    for camera_in, sonar_in in tzip(camera_in_list, sonar_in_list):
        assert (
            camera_in.name == sonar_in.name
        ), "camera and sonar file names do not match, perhaps filesystem is not loading files sequentially."

        # read images -- cam+sonar
        img_cam_in = cv2.imread(str(camera_in))
        img_sonar_in = cv2.imread(str(sonar_in))

        # darken cam
        img_cam_processed = (1 - d) * img_cam_in

        # blur cam
        img_cam_processed = cv2.GaussianBlur(
            src=img_cam_processed, ksize=kernel, sigmaX=var, sigmaY=var
        )

        # stack -- cam+processed+sonar
        img_out = cv2.hconcat(
            [img_cam_in, img_cam_processed.astype("uint8"), img_sonar_in]
        )

        # save
        cv2.imwrite(
            str(dataset_out_dir / camera_in.parent.stem / camera_in.name), img_out
        )
