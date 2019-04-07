"""
Simple example script on the OpenNeuro Haxby 2001 dataset


requires awscli: `apt install awscli`
"""
import os
import json
from time import sleep

data_uris = {
    '': 's3://openneuro.org/ds000105',
    '/derivatives': 's3://openneuro.outputs/2dc61bcfafc8ebde6841628fe0540112/d84d0800-c165-4660-9035-7af5b71b7821'
}

dataset_dir = os.path.join('data', 'ds000105')
if not os.path.isdir(dataset_dir):
    os.makedirs(dataset_dir)
    for folder, uri in sorted(data_uris.items()):
        cmd = 'aws --no-sign-request s3 sync {} data/ds000105{}'.format(uri, folder)
        print('downloading {}..'.format(folder))
        os.system(cmd)

    ## adapt dataset description to mention fmriprep
    sleep(0.1)
    desc_file_path = os.path.join(dataset_dir, 'dataset_description.json')
    with open(desc_file_path) as fh:
        desc = json.load(fh)
    desc["PipelineDescription"] = {"Name": "fmriprep"}
    sleep(0.1)
    fmriprep_desc_fpath = os.path.join(
        dataset_dir, 'derivatives', 'fmriprep', 'dataset_description.json')
    with open(fmriprep_desc_fpath, 'w') as fh:
        json.dump(desc, fh)
    sleep(0.1)
else:
    print('found data.')


## run pyGLMdenoise on our BIDS dataset:
from glmdenoise import run_bids_directory
run_bids_directory(dataset_dir)
