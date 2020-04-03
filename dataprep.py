#!/usr/bin/python
#-*- coding: utf-8 -*-
# The script downloads the VoxCeleb datasets and converts all files to WAV.
# Requirement: ffmpeg and wget running on a Linux system.

import argparse
import os
import subprocess
import pdb
import hashlib
import time
import glob
from zipfile import ZipFile
from tqdm import tqdm

## ========== ===========
## Parse input arguments
## ========== ===========
parser = argparse.ArgumentParser(description = "VoxCeleb downloader");

parser.add_argument('--save_path', 	type=str, default="data", help='Target directory');
parser.add_argument('--user', 		type=str, default="user", help='Username');
parser.add_argument('--password', 	type=str, default="pass", help='Password');

parser.add_argument('--download', dest='download', action='store_true', help='Enable download')
parser.add_argument('--extract',  dest='extract',  action='store_true', help='Enable extract')
parser.add_argument('--convert',  dest='convert',  action='store_true', help='Enable convert')

args = parser.parse_args();

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
def download(args, lines):

	for line in lines:
		url 	= line.split()[0]
		md5gt 	= line.split()[1]
		outfile = url.split('/')[-1]

		## Download files
		out 	= subprocess.call('wget %s --user %s --password %s -O %s/%s'%(url,args.user,args.password,args.save_path,outfile), shell=True)
		if out != 0:
			raise ValueError('Download failed %s. If download fails repeatedly, use alternate URL on the VoxCeleb website.'%url)

		## Check MD5
		md5ck 	= md5('%s/%s'%(args.save_path,outfile))
		if md5ck == md5gt:
			print('Checksum successful %s.'%outfile)
		else:
			raise ValueError('Checksum failed %s.'%outfile)

## ========== ===========
## Concatenate file parts
## ========== ===========
def concatenate(args,lines):

	for line in lines:
		infile 	= line.split()[0]
		outfile	= line.split()[1]
		md5gt 	= line.split()[2]

		## Concatenate files
		out 	= subprocess.call('cat %s/%s > %s/%s' %(args.save_path,infile,args.save_path,outfile), shell=True)

		## Check MD5
		md5ck 	= md5('%s/%s'%(args.save_path,outfile))
		if md5ck == md5gt:
			print('Checksum successful %s.'%outfile)
		else:
			raise ValueError('Checksum failed %s.'%outfile)

		out 	= subprocess.call('rm %s/%s' %(args.save_path,infile), shell=True)

## ========== ===========
## Extract zip files
## ========== ===========
def extract(args):

	files = glob.glob('%s/*.zip'%args.save_path)

	for fname in files:
		print('Extracting %s'%fname)
		zf = ZipFile(fname, 'r')
		zf.extractall(args.save_path)
		zf.close()

## ========== ===========
## Convert
## ========== ===========
def convert(args):

	folders = glob.glob('%s/voxceleb2/*/*/'%args.save_path)
	files 	= glob.glob('%s/voxceleb2/*/*/*.m4a'%args.save_path)
	files.sort()

	print('Converting files from AAC to WAV')
	for fname in tqdm(files):
		outfile = fname.replace('.m4a','.wav')
		out = subprocess.call('ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s >/dev/null 2>/dev/null' %(fname,outfile), shell=True)
		if out != 0:
			raise ValueError('Conversion failed %s.'%fname)

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

	if args.download:
		download(args,fileparts)

	if args.extract:
		concatenate(args, files)
		extract(args)
		out = subprocess.call('mv %s/dev/aac/* %s/aac/ && rm -r %s/dev' %(args.save_path,args.save_path,args.save_path), shell=True)
		out = subprocess.call('mv %s/wav %s/voxceleb1' %(args.save_path,args.save_path), shell=True)
		out = subprocess.call('mv %s/aac %s/voxceleb2' %(args.save_path,args.save_path), shell=True)

	if args.convert:
		convert(args)
		
