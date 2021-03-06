import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpuid', nargs="+", type=int, default=[0])  # example: "--gpuid 0 2 3"
parser.add_argument('--model_path', type=str, default='./models/',
                    help='path for saving trained models')
parser.add_argument('--model_load_path_net', type=str, default='',
                    help='path for saving trained models')
parser.add_argument('--model_load_path_relationNet', type=str, default='',
                    help='path for saving trained models')
#parser.add_argument('--model_load_path', type=str, default='',
#                    help='path for saving trained models')

parser.add_argument('--save_step', type=int, default=500,
                    help='step size for saving trained models')

# Model parameters
parser.add_argument('--num_filter', type=int, default=64,
                    help='number of filter for feature extractor')
parser.add_argument('--num_in_channel', type=int, default=3)
parser.add_argument('--num_fc', type=int, default=8)

# Experiment parameters
parser.add_argument('--batch_size', type=int, default=80)
parser.add_argument('--batch_size_test', type=int, default=40)
parser.add_argument('--crop_size', type=int, default=80)
parser.add_argument('--central_crop', type=bool, default=False)
parser.add_argument('--num_episode', type=int, default=1000)
parser.add_argument('--num_episode_test', type=int, default=600)
parser.add_argument('--way_train', type=int, default=5)
parser.add_argument('--way_test', type=int, default=5)
parser.add_argument('--num_query', type=int, default=15)
parser.add_argument('--num_workers', type=int, default=24)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--drop_prob', type=float, default=0.0)


parser.add_argument('--log_step', type=int, default=100,
                    help='step size for prining log info')
parser.add_argument('--logport', type=int, default=0)

args = parser.parse_args()
print(args)
