import argparse


def get_args():
    parser = argparse.ArgumentParser()
    # Random seed for np and tf (-1 to avoid seeding)
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    # Training parametes
    parser.add_argument('--dataset-path', type=str, default='/home/idm/Lenslet_Gscale',
                        help='Direcory with training pngs')

    parser.add_argument('--test-path', type=str, default='/scratch/dataset/jvet_1stframe/png/',
                        help='Direcory with test pngs')


    parser.add_argument('--n-crops', type=int, default=216, help='Number of crops for each image')


    # TODO we need a separate bithdepth switch for each dataset!
    #block size in terms of macro pixels, must be multiplied by number of views in lenslet format
    parser.add_argument('--context-size', type=int, default=64,
                        help='Size of the context [64, 128] (default 64x64))')
    parser.add_argument('--predictor-size', type=int, default=32,
                        help='Size of the predictor [32, 64] (default 32x32)')
    parser.add_argument('--bit-depth', type=int, default=8,
                        help='Depth of the samples, in bits per pixel (default 8)')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs to test (default: 100)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (default: 64). For crop dataloaders, teh actual BS is multiplied by crops_per_image')
    parser.add_argument('--loss', type=str, default='mse', help='Loss functionto minimize [abs|mse|ssim]')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate (default: 0.001)')
    parser.add_argument('--lr-gamma', type=float, default=0.1, help='Learning rate decay factor (default: 0.1)')
    parser.add_argument('--lr-min', type=float, default=0.0, help='Learning rate decay factor (default: 0.1)')
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-scheduler", default="exponentiallr", type=str,
                        help="the lr scheduler (default: steplr)")


    # parameters of the DataSet
    parser.add_argument("--num_views_ver",  default=9,      type=int, help="Num Views Vertical")
    parser.add_argument("--num_views_hor",  default=9,      type=int, help="Num Views Horizontally")
    parser.add_argument("--resol_ver",      default=4992,   type=int, help="Vertical Resolution")
    parser.add_argument("--resol_hor",      default=7488,   type=int, help="Horizontal Resolution")
    parser.add_argument("--bit_depth",      default=8,      type=int, help="Bit Depth")



    parser.add_argument('--save', default='../runs/exp', type=str,
                        help='Output dir')
    parser.add_argument('--project-name', default='delete', type=str)

    #@TODO automatizar resuming simulations
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--model', default='Unet2k', type=str)


    args = parser.parse_args()
    return args