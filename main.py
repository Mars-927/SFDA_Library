import argparse
import os
from methods.SHOT.train_src_shot import shot_src
from methods.SHOT.train_tar_shot import shot_tar

from utils.Other import seed_everything
from utils.Project_Record import Project
from utils.Dataset import get_dataloader_select
from utils.Evaluate import test_target_shot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--name', type=str, default="source test")
    parser.add_argument('--dataset', type=str, default="OFFICE31",  choices = ["AID_NWPU_UCM","OFFICE31"])
    parser.add_argument('--method', type=str, default="source_shot")
    parser.add_argument('--gpu_id', type=str, default='1')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    seed_everything(args.seed)
    Project(args.name)
    
    methods = {
        'source':None,
        'source_shot': shot_src,
        'shot': shot_tar,
        'guidingPseudoSFDA':None,
    }

    if args.dataset == "AID_NWPU_UCM":
        args.num_class = 10
        args.dataset_path = "dataset/AID_NWPU_UCM"
        args.image_root = "F:/HANS/!dataset/RS_DomainAdaptation_AIDUCMNWPU"
        args.domains = ["AID", "NWPU-RESISC45", "UCMerced_LandUse"]

    elif args.dataset == "OFFICE31":
        args.class_num = 31
        args.dataset_path = "dataset/OFFICE31"
        args.image_root = "F:/HANS/!dataset/OFFICE31"
        args.domains = ["amazon", "dslr", "webcam"]

    elif args.dataset == "office_home":
        args.class_num = 65
        args.dataset_path = "dataset/OfficeHomeDataset"
        args.image_root = "F:/HANS/!dataset/OfficeHomeDataset"
        args.domains = ["Art", "Clipart", "Product", "Real World"]
    else:
        assert False,"unknown dataset"

    assert args.method in methods.keys(), "unknown method"
    method = args.method
    method_func = methods[method]
    if "source" in method:
        for source_domain in args.domains:
            args.source_domain = source_domain
            Project.log(f"Train Source; Dataset:{args.dataset}; Domain {source_domain}")
            dataset_dirt = get_dataloader_select(args, source_domain)
            method_func(args, dataset_dirt)
            for target_domain in args.domains:
                if target_domain == source_domain:
                    continue
                dataset_dirt = get_dataloader_select(args, target_domain)
                acc = test_target_shot(args, dataset_dirt['all'], Project.root_path)
                Project.log(f"Test Target; Dataset:{args.dataset}; {source_domain} => {target_domain}; acc: {acc:.4f}")
                
    else:
        for target_domain in args.domains:
            dataset_dirt = get_dataloader_select(args, target_domain)
            for source_domain in args.domains:
                if target_domain == source_domain:
                    continue
                Project.log(f"Domain Adaptation: {args.name} {source_domain} => {target_domain}")
                args.target_domain = target_domain
                args.source_domain = source_domain
                method_func(args, dataset_dirt)
