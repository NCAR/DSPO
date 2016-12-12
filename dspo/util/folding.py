import os
import csv

def parse_foldingcsv(rawdatapath):
    # data dict
    rawdata = {}
    kernel = os.path.basename(rawdatapath).split('.')[0]

    # read csv file
    with open(rawdatapath, 'r') as cf:
        hwcsv = csv.reader(cf, delimiter=';')
        for kname, dummy, ename, tick, hwcnt in hwcsv:
            if kname == kernel and "_per_ins" not in ename:
                if ename not in rawdata:
                    rawdata[ename] = []
                try:
                    rawdata[ename].append(float(hwcnt))
                except:
                    rawdata[ename].append(float('nan'))

    # remove unstable data
    data = {}
    for ename, hwc in rawdata.items():
        data[ename] = hwc[50:-50]

    return data
