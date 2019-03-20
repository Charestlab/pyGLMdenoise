"""
Simple example script on the OpenNeuro Haxby 2001 dataset


requires awscli: `apt install awscli`
"""
import os

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
else:
    print('found data.')
