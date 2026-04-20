#!/usr/bin/python
#-*- coding: utf-8 -*-
# The script downloads the VoxCeleb datasets and converts all files to WAV.
# Requirement: ffmpeg and wget running on a Linux system.

import argparse
import os
import shutil
import subprocess
import hashlib
import glob
import tarfile
from zipfile import ZipFile
from tqdm import tqdm
from scipy.io import wavfile

## ========== ===========
## Parse input arguments
## ========== ===========
parser = argparse.ArgumentParser(description = "VoxCeleb downloader")

parser.add_argument('--save_path',     type=str, default="data", help='Target directory')
parser.add_argument('--key',          type=str, default="",     help='Access key for VoxCeleb download')

parser.add_argument('--download',    dest='download',    action='store_true', help='Enable download')
parser.add_argument('--concatenate', dest='concatenate', action='store_true', help='Concatenate downloaded file parts')
parser.add_argument('--extract',     dest='extract',     action='store_true', help='Extract downloaded files')
parser.add_argument('--convert',     dest='convert',     action='store_true', help='Enable convert')
parser.add_argument('--augment',     dest='augment',     action='store_true', help='Download and extract augmentation files')

args = parser.parse_args()

## ========== ===========
## MD5SUM
## ========== ===========
def md5(fname):

    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

## ========== ===========
## Download with wget
## ========== ===========
def download(args, lines, use_key=False):

    for line in lines:
        filename = line.split()[0]
        md5gt    = line.split()[1]

        if use_key:
            ## VoxCeleb download with access key
            url     = f'https://cn01.mmai.io/download/voxceleb?file={filename}&key={args.key}'
            outfile = filename
        else:
            ## Direct URL (e.g. augmentation files)
            url     = filename
            outfile = url.split('/')[-1]

        outpath = f'{args.save_path}/{outfile}'

        ## Skip if file already exists with correct checksum
        if os.path.exists(outpath):
            if md5(outpath) == md5gt:
                print(f'Skipping {outfile}, already exists with correct checksum.')
                continue

        ## Download files (wget -O overwrites existing files)
        subprocess.run(['wget', '--no-check-certificate', url, '-O', outpath], check=True)

        ## Verify checksum
        md5ck     = md5(outpath)
        if md5ck == md5gt:
            print(f'Checksum successful {outfile}.')
        else:
            raise ValueError(f'Checksum failed {outfile}.')

## ========== ===========
## Concatenate file parts
## ========== ===========
def concatenate(args,lines):

    for line in lines:
        infile     = line.split()[0]
        outfile    = line.split()[1]
        md5gt     = line.split()[2]

        ## Concatenate file parts
        parts = sorted(glob.glob(f'{args.save_path}/{infile}'))
        with open(f'{args.save_path}/{outfile}', 'wb') as outf:
            for part in parts:
                with open(part, 'rb') as inf:
                    shutil.copyfileobj(inf, outf)

        ## Check MD5
        md5ck     = md5(f'{args.save_path}/{outfile}')
        if md5ck == md5gt:
            print(f'Checksum successful {outfile}.')
        else:
            raise ValueError(f'Checksum failed {outfile}.')

        ## Remove file parts
        for part in parts:
            os.remove(part)

## ========== ===========
## Extract zip files
## ========== ===========
def is_within_directory(directory, target):
    
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)

    prefix = os.path.commonprefix([abs_directory, abs_target])
    
    return prefix == abs_directory

def safe_extract(tar, path=".", members=None, *, numeric_owner=False):

    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not is_within_directory(path, member_path):
            raise Exception("Attempted Path Traversal in Tar File")
    tar.extractall(path, members, numeric_owner=numeric_owner)

def full_extract(args, fname, extract_path=None):

    if extract_path is None:
        extract_path = args.save_path

    print(f'Extracting {fname} to {extract_path}')
    os.makedirs(extract_path, exist_ok=True)
    if fname.endswith(".tar.gz"):
        with tarfile.open(fname, "r:gz") as tar:
            safe_extract(tar, extract_path)
    elif fname.endswith(".zip"):
        with ZipFile(fname, 'r') as zf:
            zf.extractall(extract_path)


## ========== ===========
## Partially extract zip files
## ========== ===========
def part_extract(args, fname, target):

    print(f'Extracting {fname}')
    with ZipFile(fname, 'r') as zf:
        for infile in zf.namelist():
            if any([infile.startswith(x) for x in target]):
                zf.extract(infile,args.save_path)
            # pdb.set_trace()
            # zf.extractall(args.save_path)

## ========== ===========
## Convert
## ========== ===========
def convert(args):

    files     = glob.glob(f'{args.save_path}/voxceleb2/*/*/*/*/*.m4a')
    files.sort()

    print('Converting files from AAC to WAV')
    for fname in tqdm(files):
        outfile = fname.replace('/aac/', '/wav/').replace('.m4a', '.wav')
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        subprocess.run(['ffmpeg', '-y', '-i', fname, '-ac', '1', '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', outfile],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

## ========== ===========
## Split MUSAN for faster random access
## ========== ===========
def split_musan(args):

    files = glob.glob(f'{args.save_path}/musan/*/*/*.wav')

    audlen = 16000*5
    audstr = 16000*3

    print('Splitting MUSAN for faster random access')
    for file in tqdm(files):
        fs,aud = wavfile.read(file)
        writedir = os.path.splitext(file.replace('/musan/','/musan_split/'))[0]
        os.makedirs(writedir)
        for st in range(0,len(aud)-audlen,audstr):
            wavfile.write(f'{writedir}/{st/fs:05.0f}.wav',fs,aud[st:st+audlen])

## ========== ===========
## Main script
## ========== ===========
if __name__ == "__main__":

    if not os.path.exists(args.save_path):
        raise ValueError('Target directory does not exist.')

    f = open('lists/fileparts.txt','r')
    fileparts = f.readlines()
    f.close()

    f = open('lists/files.txt','r')
    files = f.readlines()
    f.close()

    f = open('lists/augment.txt','r')
    augfiles = f.readlines()
    f.close()

    if args.augment:
        download(args, augfiles)
        part_extract(args,os.path.join(args.save_path,'rirs_noises.zip'),['RIRS_NOISES/simulated_rirs/mediumroom','RIRS_NOISES/simulated_rirs/smallroom'])
        full_extract(args,os.path.join(args.save_path,'musan.tar.gz'))
        split_musan(args)

    if args.download:
        download(args, fileparts, use_key=True)

    if args.concatenate:
        concatenate(args, files)

    if args.extract:
        ## Extract VoxCeleb1 dev (from concatenated zip) and test
        full_extract(args, os.path.join(args.save_path, 'vox1_dev_wav.zip'),
                     os.path.join(args.save_path, 'voxceleb1', 'dev'))
        full_extract(args, os.path.join(args.save_path, 'vox1_test_wav.zip'),
                     os.path.join(args.save_path, 'voxceleb1', 'test'))

        ## Extract VoxCeleb2 dev (from concatenated zip) and test
        full_extract(args, os.path.join(args.save_path, 'vox2_dev_aac.zip'),
                     os.path.join(args.save_path, 'voxceleb2'))
        full_extract(args, os.path.join(args.save_path, 'vox2_test_aac.zip'),
                     os.path.join(args.save_path, 'voxceleb2', 'test'))

    if args.convert:
        convert(args)
        
