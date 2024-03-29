import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='/disks/disk0/private/ljk/sock_project/sock_classification/data_classification/sock_classification')
parser.add_argument('--n_obj', type=int, default=9)
parser.add_argument('--n_round', type=int, default=1)
parser.add_argument('--n_train_per_obj', type=int, default=4000)
parser.add_argument('--n_valid_per_obj', type=int, default=500)
parser.add_argument('--n_test_per_obj', type=int, default=1000)
parser.add_argument('--calibrate', type=int, default=0)

parser.add_argument('--input_window_size', type=int, default=45)
parser.add_argument('--skip', type=int, default=2)
parser.add_argument('--subsample', type=int, default=1)

parser.add_argument('-epoch','--n_epoch', type=int, default=60)
parser.add_argument('-extra_data','--extra_data', type=int, default=0)
parser.add_argument('-l1','--lambda_L1', type=float, default=0)
parser.add_argument('-lr','--lr', type=float, default=1e-3)
parser.add_argument('-l2','--weight_decay', type=float, default=1e-4)
parser.add_argument('-batch','--batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('-seed','--seed', type=int, default=10)

parser.add_argument('--log_per_iter', type=int, default=10000)

parser.add_argument('-depth','--num_hidden_layers', type=int, default=6)
parser.add_argument('-mask','--mask_ratio', type=float, default=0.1)
parser.add_argument('-n_pretrain','--n_pretrain', type=int, default=60)
parser.add_argument('-mask_type','--mask_type', type=str, default='tube')
parser.add_argument('-tubelet_size','--tubelet_size', type=int, default=5)
parser.add_argument('-patch_size','--patch_size', type=int, default=4)
parser.add_argument('-pretrain','--pretrain', type=int, default=1)
parser.add_argument('-encoder_dims','--hidden_size', type=int, default=768)
parser.add_argument('-encoder_att_heads','--encoder_att_heads', type=int, default=12)
parser.add_argument('-comparisons','--comparisons', type=int, default=30)
parser.add_argument('-beta','--beta', type=float, default=0.5)
parser.add_argument('-lr_pretrain','--lr_pretrain',type=float,default=0.005)

def gen_args(model="ViT"):
    args, unknown = parser.parse_known_args()

    if args.n_obj == 9:
        args.object_list = [
            'downstairs', 'jump', 'lean_left', 'lean_right',
            'stand', 'stand_toes', 'upstairs', 'walk', 'walk_fast']
        args.n_rounds = [24, 3, 3, 3, 3, 4, 24, 3, 3]
    else:
        raise AssertionError("Unknown number of classes %d" % args.n_obj)


    args.rec_path = 'dump_vest_classification_nObj_%d_subsample_%d' % (
        args.n_obj, args.subsample)

    os.system('mkdir -p ' + args.rec_path)

    return args
