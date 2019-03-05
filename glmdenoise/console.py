import argparse


def main():
    parser = argparse.ArgumentParser(prog='glmdenoise')
    parser.add_argument('directory', nargs='?', default='.', 
        help='Data directory containing BIDS.')
    parser.add_argument('--subject', default=None, 
        help='Subject number.')
    args = parser.parse_args()
    print(args.directory)
    print(args.subject)
