__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2019/06/24 19:19:51"

import time
num_beads = 64 #1000

file_handle = open("./sample1.psf", 'w')

## start line
print("PSF CMAP CHEQ XPLOR", file = file_handle)
print("", file = file_handle)

## title
print("{:>8d} !NTITLE".format(2), file = file_handle)
print("* HOMOPOLYMER PSF FILE", file = file_handle)
print("* DATE: {} CREATED BY USER: DINGXQ".format(time.asctime()), file = file_handle)

## atoms
print("", file = file_handle)
print("{:>8d} !NATOM".format(num_beads), file = file_handle)
for i in range(1, num_beads+1):
    print("{:>8d} {:<4s} {:<4d} {:<4s} {:<4s} {:<4s} {:<14.6}{:<14.6}{:>8d}{:14.6}".format(i, "POL", 1, "POL", "C", "C", 0.0, 0.0, 0, 0.0), file = file_handle)

## bonds
print("", file = file_handle)
print("{:>8d} !NBOND: bonds".format(num_beads-1), file = file_handle)
i = 1
count = 0
while i < num_beads:
    print("{:>8d}{:>8d}".format(i,i+1), end = "", file = file_handle)
    count += 1
    if count == 4:
        print("", file = file_handle)
        count = 0
    i += 1
print("", file = file_handle)

## angles
print("", file = file_handle)
print("{:>8d} !NTHETA: angles".format(0), file = file_handle)
print("", file = file_handle)

## diherals
print("", file = file_handle)
print("{:>8d} !NPHI: dihedrals".format(0), file = file_handle)
print("", file = file_handle)

## impropers
print("", file = file_handle)
print("{:>8d} !NIMPHI: impropers".format(0), file = file_handle)
print("", file = file_handle)
file_handle.close()
