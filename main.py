import argparse
import os

from methods.guidingPseudoSFDA.train_tar_guidingPseudoSFDA import guidingPseudoSFDA_tar
from methods.SHOT.train_src_shot import shot_src
from methods.SHOT.train_tar_shot import shot_tar
from methods.NRC.train_tar_nrc import nrc_tar
from methods.AaD.train_tar_AaD import AaD_tar
from methods.SFDA.train_tar_sfda import SFDA_tar
from methods.GSFDA.train_src_gsfda import gsfda_src
from utils.Dataset import get_dataloader_select
from utils.Evaluate import test_shot_source, test_gsfda_source
from utils.Other import seed_everything
from utils.Project_Record import Project

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--name', type=str, default="GSFDA_AID_NWPU_UCM_Target")                                               # create folder name
    parser.add_argument('--pretrain', type=str, default="GSFDA_AID_NWPU_UCM_Source")                                           # pretrain model folder name
    parser.add_argument('--mode', type=str, default="Source", choices=["Source","Target"])             # Pretrain or domain domain
    parser.add_argument('--dataset', type=str, default="AID_NWPU_UCM", choices = ["AID_NWPU_UCM","OFFICE31"])       # AID_NWPU_UCM Base office31
    parser.add_argument('--method', type=str, default="gsfda")
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    seed_everything(args.seed)
    Project(args.name)

    # check
    methods = {
        'source_shot': shot_src,
        'shot': shot_tar,                               # 2020
        'nrc': nrc_tar,                                 # 2021
        'gsfda':gsfda_src,                              # 2021
        'AaD': AaD_tar,                                 # 2022
        'CSFDA': None,                                  # 2023
        'guidingPseudoSFDA': guidingPseudoSFDA_tar,     # 2023
        'SFDA':SFDA_tar,
    }
    evals = {
        'source_shot': test_shot_source,
        'shot': test_shot_source,
        'nrc': test_shot_source,
        'gsfda':test_gsfda_source,
        'AaD': test_shot_source,
        'CSFDA': None,
        'guidingPseudoSFDA': test_shot_source,
        'SFDA':test_shot_source,
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
    else:
        assert False,"unknown dataset"
    assert args.method in methods.keys(), "unknown method"


    method = args.method
    transfer_method = methods[method]
    eval_method = evals[method]    

    # pretrain source
    if args.mode == "Source":
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

# python .\main.py --name guidingPseudoSFDA_AID_NWPU_UCM_Target --pretrain SHOT_AID_NWPU_UCM_Source --dataset AID_NWPU_UCM --gpu_id 1 --method guidingPseudoSFDA
# python .\main.py --name SFDA_AID_NWPU_UCM_Target --pretrain SHOT_AID_NWPU_UCM_Source --dataset AID_NWPU_UCM --gpu_id 1 --method SFDA