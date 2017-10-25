import argparse
import os
from evaluation import generate_pr

def run_generate_pr(args):
    # (Algorithm slug, Algorithm ID) pairs for each
    # intrinsic image decomposition algorithm we want to evaluate
    algos = [
        ('baseline_reflectance', 759),
        ('shen2011_optimization', 426),
        ('grosse2009_color_retinex', 633),
        ('garces2012_clustering', 1221),
        ('zhao2012_nonlocal', 709),
        ('bell2014_densecrf', 1141),
        ('zhou2015_reflprior', 1281),
        ('lettry2018_siamese', 1402),
    ]
    algo_configs = [
        (algo_slug, os.path.join(args.decomp_dir, '%s-%s' % (algo_slug, algo_id)))
        for algo_slug, algo_id in algos
    ]
    generate_pr(
        saw_image_dir=args.saw_image_dir,
        pixel_labels_dir=args.pixel_labels_dir,
        splits_dir=args.splits_dir,
        out_dir=args.out_dir,
        dataset_split=args.dataset_split,
        class_weights=[1, 1, 2],
        bl_filter_size=10,
        algo_configs=algo_configs,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', help='Which command to run. Currently we support "generate_pr".')
    parser.add_argument('--pixel_labels_dir', default='saw/saw_pixel_labels/saw_data-filter_size_0-ignore_border_0.05-normal_gradmag_thres_1.5-depth_gradmag_thres_2.0', help='Path to the directory which contains the ground truth labels.')
    parser.add_argument('--saw_image_dir', default='saw/saw_images_512', help='Path to the directory which contains the SAW photos.')
    parser.add_argument('--saw_anno_dir', default='saw/saw_annotations_json', help='Path to the directory which contains the SAW annotations.')
    parser.add_argument('--nyu_dataset_dir', default='saw/saw_nyu-depthv2', help='Path to the directory which contains the NYUv2 dataset normal and depth maps.')
    parser.add_argument('--splits_dir', default='saw/saw_splits', help='Path to the directory which contains the dataset splits.')
    parser.add_argument('--net_dir', default='saw/saw_pixelnet-614', help='Path to the directory which contains our trained CNN.')
    parser.add_argument('--decomp_dir', default='saw/saw_decomps', help='Path to the directory which contains the decompositions of the intrinsic image algorithms.')
    parser.add_argument('--out_dir', default='out_dir', help='Path to the directory where we will save the generated precision-recall curves.')
    parser.add_argument('--dataset_split', default='E', help='Which dataset split we should use: "R" (training), "V" (validation), "E" (test).')
    args = parser.parse_args()

    if args.command == 'generate_pr':
        run_generate_pr(args)
    else:
        raise ValueError('Unsupported command: "%s"!' % args.command)


if __name__ == '__main__':
    main()
