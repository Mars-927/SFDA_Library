import argparse
import os

from methods.guidingPseudoSFDA.train_tar_guidingPseudoSFDA import guidingPseudoSFDA_tar
from methods.SHOT.train_src_shot import shot_src
from methods.SHOT.train_tar_shot import shot_tar
from methods.NRC.train_tar_nrc import nrc_tar
from methods.AaD.train_tar_AaD import AaD_tar
from utils.Dataset import get_dataloader_select
from utils.Evaluate import test_shot_source
from utils.Other import seed_everything
from utils.Project_Record import Project

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--name', type=str, default="nrc_test")  # create folder name
    parser.add_argument('--pretrain', type=str, default="source test")  # read folde name
    parser.add_argument('--dataset', type=str, default="OFFICE31",  choices = ["AID_NWPU_UCM","OFFICE31"])
    parser.add_argument('--method', type=str, default="nrc")
    parser.add_argument('--gpu_id', type=str, default='1')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    seed_everything(args.seed)
    Project(args.name)
    
    
    
    methods = {
        'source_shot': shot_src,
        'source_shotplus': None,
        'shot': shot_tar,
        'nrc': nrc_tar,
        'shotplus': None,
        'AaD': AaD_tar,
        'CSFDA': None,
        'guidingPseudoSFDA':None,
    }
    evals = {
        'source_shot': test_shot_source,
        'source_shotplus': test_shot_source,
        'shot': test_shot_source,
        'nrc': test_shot_source,
        'shotplus': test_shot_source,
        'AaD': test_shot_source,
        'CSFDA': None,
        'guidingPseudoSFDA':None,
    }


    if args.dataset == "AID_NWPU_UCM":
        args.class_num = 10
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
    transfer_method = methods[method]
    eval_method = evals[method]
    

    # pretrain source
    if "source" in method:
        for source_domain in args.domains:
            args.source_domain = source_domain
            log_str = f"Train Source; Dataset:{args.dataset}; Domain {source_domain};\n"
            Project.log(log_str)
            dataset_dirt = get_dataloader_select(args, source_domain)
            transfer_method(args, dataset_dirt)

            # eval only source
            for target_domain in args.domains:
                if target_domain == source_domain:
                    continue
                dataset_dirt = get_dataloader_select(args, target_domain)
                acc = eval_method(args, dataset_dirt['all'], Project.root_path,  source_domain)
                log_str = f"\nTest Target; Dataset:{args.dataset}; {source_domain} => {target_domain}; acc: {acc:.4f}\n\n\n"
                Project.log(log_str, "score.txt")

    # domain adaptation
    else:
        assert os.path.exists(f"process/{args.pretrain}"), "empty pretrain"
        args.weight_basepath = f"process/{args.pretrain}"
        for source_domain in args.domains:
            args.source_domain = source_domain

            # Domain adaptation
            for target_domain in args.domains:
                args.target_domain = target_domain
                if target_domain == source_domain:
                    continue
                dataset_dirt = get_dataloader_select(args, target_domain)
                log_str = f"Domain Adaptation: {args.name} {source_domain} => {target_domain}\n"
                Project.log(log_str)
                transfer_method(args, dataset_dirt)

                acc = eval_method(args, dataset_dirt['test'], Project.root_path, f"{source_domain}2{target_domain}")
                log_str = f"Test Target; Method:{method} Dataset:{args.dataset}; {source_domain} => {target_domain}; acc: {acc:.4f}\n\n\n"
                Project.log(log_str, "score.txt")

