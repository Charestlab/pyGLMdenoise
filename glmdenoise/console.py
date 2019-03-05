import argparse


def main():
    """This function gets called when the user executes `glmdenoise`.

    It defines and interprets the console arguments, then calls
    the relevant python code.
    """

    parser = argparse.ArgumentParser(prog='glmdenoise')
    parser.add_argument('directory', nargs='?', default='.', 
        help='Data directory containing BIDS.')
    parser.add_argument('--subject', default=None, 
        help='Subject number. If not specified, will run for each subject.')
    parser.add_argument('--task', default=None, 
        help='Task name. If not specified, will run on all tasks.')
    args = parser.parse_args()
    print(args.directory)
    print(args.subject)
