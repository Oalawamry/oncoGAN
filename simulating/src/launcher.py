#!/usr/local/bin/python3

import warnings
warnings.filterwarnings(action="ignore", message="h5py is running against HDF5")

import click
from oncoGAN import availTumors, oncoGAN
from mergeVCFs import mergeVCFs

@click.group()
def cli():
    pass

cli.add_command(availTumors)
cli.add_command(oncoGAN)
cli.add_command(mergeVCFs)
if __name__ == '__main__':
    cli()
