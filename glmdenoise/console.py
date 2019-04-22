import argparse
import os
from glmdenoise.io.directory import run_bids_directory
from glmdenoise.io.public import run_public


def main():
    """This function gets called when the user executes `glmdenoise`.

    It defines and interprets the console arguments, then calls
    the relevant python code.
    """

    parser = argparse.ArgumentParser(prog='glmdenoise')
    parser.add_argument('dataset', nargs='?', default='.', 
        help='Data directory containing BIDS, or name of public dataset.')
    parser.add_argument('--subject', default=None, 
        help='Subject number. If not specified, will run for each subject.')
    parser.add_argument('--task', default=None, 
        help='Task name. If not specified, will run on all tasks.')
    args = parser.parse_args()
    if os.path.isdir(args.dataset):
        run_bids_directory(args.dataset, args.subject, args.task)
    else:
        run_public(args.dataset, args.subject, args.task)
