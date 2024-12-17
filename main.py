import logging
import os
import sys

import click
import numpy as np
import torch

sys.path.append("src")
import backbones
import common
import metrics
import simplenet
import utils

LOGGER = logging.getLogger(__name__)

_DATASETS = {
    "mvtec": ["datasets.mvtec", "MVTecDataset"],
}


def run(args):
    run_save_path = utils.create_storage_folder(
        args.results_path, args.log_project, args.log_group, args.run_name, mode="overwrite"
    )

    pid = os.getpid()
    list_of_dataloaders = dataset(args)["get_dataloaders"](args.seed)

    device = utils.set_torch_device(args.gpu)

    result_collect = []
    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["training"].name,
                dataloader_count + 1,
                len(list_of_dataloaders),
            )
        )

        utils.fix_seeds(args.seed, device)

        dataset_name = dataloaders["training"].name

        imagesize = dataloaders["training"].dataset.imagesize
        simplenet_list = net(args)["get_simplenet"](imagesize, device)

        models_dir = os.path.join(run_save_path, "models")
        os.makedirs(models_dir, exist_ok=True)
        for i, SimpleNet in enumerate(simplenet_list):
            # torch.cuda.empty_cache()
            if SimpleNet.backbone.seed is not None:
                utils.fix_seeds(SimpleNet.backbone.seed, device)
            LOGGER.info(
                "Training models ({}/{})".format(i + 1, len(simplenet_list))
            )
            # torch.cuda.empty_cache()

            SimpleNet.set_model_dir(os.path.join(models_dir, f"{i}"), dataset_name)
            args.test = True
            if not args.test:
                i_auroc, p_auroc, pro_auroc = SimpleNet.train(dataloaders["training"], dataloaders["testing"])
            else:
                # BUG: the following line is not using. Set test with True by default.
                args.save_segmentation_images = True
                i_auroc, p_auroc, pro_auroc = SimpleNet.test(dataloaders["training"], dataloaders["testing"], args.save_segmentation_images)
                print("Warning: Pls set test with true by default")

            result_collect.append(
                {
                    "dataset_name": dataset_name,
                    "instance_auroc": i_auroc,  # auroc,
                    "full_pixel_auroc": p_auroc,  # full_pixel_auroc,
                    "anomaly_pixel_auroc": pro_auroc,
                }
            )

            for key, item in result_collect[-1].items():
                if key != "dataset_name":
                    LOGGER.info("{0}: {1:3.3f}".format(key, item))

        LOGGER.info("\n\n-----\n")

    # Store all results and mean scores to a csv-file.
    result_metric_names = list(result_collect[-1].keys())[1:]
    result_dataset_names = [results["dataset_name"] for results in result_collect]
    result_scores = [list(results.values())[1:] for results in result_collect]
    utils.compute_and_store_final_results(
        run_save_path,
        result_scores,
        column_names=result_metric_names,
        row_names=result_dataset_names,
    )

def net(args):
    backbone_names = list(args.backbone_names)
    if len(backbone_names) > 1:
        layers_to_extract_from_coll = [[] for _ in range(len(backbone_names))]
        for layer in args.layers_to_extract_from:
            idx = int(layer.split(".")[0])
            layer = ".".join(layer.split(".")[1:])
            layers_to_extract_from_coll[idx].append(layer)
    else:
        layers_to_extract_from_coll = [args.layers_to_extract_from]

    def get_simplenet(input_shape, device):
        simplenets = []
        for backbone_name, layers_to_extract_from in zip(
                backbone_names, layers_to_extract_from_coll
        ):
            backbone_seed = None
            if ".seed-" in backbone_name:
                backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(
                    backbone_name.split("-")[-1]
                )
            backbone = backbones.load(backbone_name)
            backbone.name, backbone.seed = backbone_name, backbone_seed

            simplenet_inst = simplenet.SimpleNet(device)
            simplenet_inst.load(
                backbone=backbone,
                layers_to_extract_from=layers_to_extract_from,
                device=device,
                input_shape=input_shape,
                pretrain_embed_dimension=args.pretrain_embed_dimension,
                target_embed_dimension=args.target_embed_dimension,
                patchsize=args.patchsize,
                embedding_size=args.embedding_size,
                meta_epochs=args.meta_epochs,
                aed_meta_epochs=args.aed_meta_epochs,
                gan_epochs=args.gan_epochs,
                noise_std=args.noise_std,
                dsc_layers=args.dsc_layers,
                dsc_hidden=args.dsc_hidden,
                dsc_margin=args.dsc_margin,
                dsc_lr=args.dsc_lr,
                auto_noise=args.auto_noise,
                train_backbone=args.train_backbone,
                cos_lr=args.cos_lr,
                pre_proj=args.pre_proj,
                proj_layer_type=args.proj_layer_type,
                mix_noise=args.mix_noise,
            )
            simplenets.append(simplenet_inst)
        return simplenets

    return {"get_simplenet": get_simplenet}

def dataset(args):
    dataset_info = _DATASETS[args.name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders(seed):
        dataloaders = []
        for subdataset in args.subdatasets:
            train_dataset = dataset_library.__dict__[dataset_info[1]](
                args.data_path,
                classname=subdataset,
                resize=args.resize,
                train_val_split=args.train_val_split,
                imagesize=args.imagesize,
                split=dataset_library.DatasetSplit.TRAIN,
                seed=seed,
                rotate_degrees=args.rotate_degrees,
                translate=args.translate,
                brightness_factor=args.brightness,
                contrast_factor=args.contrast,
                saturation_factor=args.saturation,
                gray_p=args.gray,
                h_flip_p=args.hflip,
                v_flip_p=args.vflip,
                scale=args.scale,
                augment=args.augment,
            )

            test_dataset = dataset_library.__dict__[dataset_info[1]](
                args.data_path,
                classname=subdataset,
                resize=args.resize,
                imagesize=args.imagesize,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed,
            )

            LOGGER.info(f"Dataset: train={len(train_dataset)} test={len(test_dataset)}")

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True, #
                num_workers=args.num_workers,
                prefetch_factor=2,
                pin_memory=True,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                prefetch_factor=2,
                pin_memory=True,
            )

            train_dataloader.name = args.name
            if subdataset is not None:
                train_dataloader.name += "_" + subdataset

            if args.train_val_split < 1:
                val_dataset = dataset_library.__dict__[dataset_info[1]](
                    args.data_path,
                    classname=subdataset,
                    resize=args.resize,
                    train_val_split=args.train_val_split,
                    imagesize=args.imagesize,
                    split=dataset_library.DatasetSplit.VAL,
                    seed=seed,
                )

                val_dataloader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    prefetch_factor=4,
                    pin_memory=True,
                )
            else:
                val_dataloader = None
            dataloader_dict = {
                "training": train_dataloader,
                "validation": val_dataloader,
                "testing": test_dataloader,
            }

            dataloaders.append(dataloader_dict)
        return dataloaders

        return {"get_dataloaders": get_dataloaders}

    return {"get_dataloaders": get_dataloaders}



import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Command line arguments')
    parser.add_argument("--results_path", type=str, default='results')
    parser.add_argument("--gpu", nargs='+', type=int, default=[0], help="GPU indices")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--log_group", type=str, default="simplenet_mvtec", help="Log group name")
    parser.add_argument("--log_project", type=str, default="MVTecAD_Results", help="Log project name")
    parser.add_argument("--run_name", type=str, default="run", help="Name of the run")
    parser.add_argument("--test", action='store_true', help="Enable test mode")
    parser.add_argument("--save_segmentation_images", action='store_true', help="Save segmentation images")

    # net command
    parser.add_argument("--backbone_names", "-b", type=str, nargs="+", default=['wideresnet50'], help="Backbone names") # resnet18, resnet50, resnet101, resnext101, wideresnet50, wideresnet101
    parser.add_argument("--layers_to_extract_from", "-le", type=str, nargs="+", default=['layer2', 'layer3'], help="Layers to extract features from") # 'layer1', 'layer2', 'layer3', 'layer4'
    parser.add_argument("--pretrain_embed_dimension", type=int, default=1536, help="Dimension of pre-trained embeddings")
    parser.add_argument("--target_embed_dimension", type=int, default=1536, help="Dimension of target embeddings")
    parser.add_argument("--patchsize", type=int, default=3, help="Patch size for Simplenet")
    parser.add_argument("--embedding_size", type=int, default=256, help="Size of the embedding")
    parser.add_argument("--meta_epochs", type=int, default=40, help="Number of meta-epochs")
    parser.add_argument("--aed_meta_epochs", type=int, default=1, help="Number of autoencoder discriminator meta-epochs")
    parser.add_argument("--gan_epochs", type=int, default=4, help="Number of GAN epochs")
    parser.add_argument("--dsc_layers", type=int, default=2, help="Number of discriminator layers")
    parser.add_argument("--dsc_hidden", type=int, default=1024, help="Hidden units in the discriminator")
    parser.add_argument("--noise_std", type=float, default=0.015, help="Standard deviation of noise")
    parser.add_argument("--dsc_margin", type=float, default=0.5, help="Margin for the discriminator")
    parser.add_argument("--dsc_lr", type=float, default=0.0002, help="Learning rate of the discriminator")
    parser.add_argument("--auto_noise", type=float, default=0, help="Auto noise")
    parser.add_argument("--train_backbone", action='store_true', help="Train backbone")
    parser.add_argument("--cos_lr", action='store_true', help="Use cosine learning rate")
    parser.add_argument("--pre_proj", type=int, default=1, help="Pre-projection")  # 是否需要适配
    parser.add_argument("--proj_layer_type", type=int, default=0, help="Projection layer type")
    parser.add_argument("--mix_noise", type=int, default=1, help="Mix noise")

    # dataset command
    parser.add_argument("--name", type=str, default='mvtec', help="Name of the dataset")
    parser.add_argument("--data_path", type=str, default='G:/Anomaly/Dataset/MVTec', help="Path to the dataset")
    # parser.add_argument("--subdatasets", "-d", type=str, nargs="+", default=['screw', 'pill', 'capsule', 'carpet', 'grid', 'tile', 'wood', 'zipper', 'cable', 'toothbrush', 'transistor', 'metal_nut', 'bottle', 'hazelnut', 'leather'], help="Subdatasets")
    parser.add_argument("--subdatasets", "-d", type=str, nargs="+", default=['zipper'], help="Subdatasets")
    parser.add_argument("--train_val_split", type=float, default=1, help="Train validation split ratio")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size") #8
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers")
    parser.add_argument("--resize", type=int, default=329, help="Size to resize images to")
    parser.add_argument("--imagesize", type=int, default=288, help="Size of the images")
    parser.add_argument("--rotate_degrees", type=int, default=0, help="Degree of rotation")
    parser.add_argument("--translate", type=float, default=0, help="Translation factor")
    parser.add_argument("--scale", type=float, default=0.0, help="Scaling factor")
    parser.add_argument("--brightness", type=float, default=0.0, help="Brightness factor")
    parser.add_argument("--contrast", type=float, default=0.0, help="Contrast factor")
    parser.add_argument("--saturation", type=float, default=0.0, help="Saturation factor")
    parser.add_argument("--gray", type=float, default=0.0, help="Grayscale factor")
    parser.add_argument("--hflip", type=float, default=0.0, help="Horizontal flip probability")
    parser.add_argument("--vflip", type=float, default=0.0, help="Vertical flip probability")
    parser.add_argument("--augment", action='store_true', help="Enable data augmentation")
    args = parser.parse_args()

    run(args)


