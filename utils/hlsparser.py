#!/usr/bin/python

# HLS report parser.
# Author: Tiago Lascasas dos Santos
# Available on Github: https://github.com/tiagolascasas/Vivado-HLS-Report-Parser/blob/master/hlsparser.py

import sys
import csv
import xml.etree.ElementTree as ET
from os import path

usage = '''
Usage:
\thlsparser <path to report> <name of input code> <optimizations>
\t - <path to report>: Relative path to the csynth.xml file (including the file)
\t                     Alternatively, use -d to assume default location
\t - <name of the input code>: Benchmark name, with commas if it has spaces
\t - <optimizations>: Optimizations performed, with commas if it has spaces
'''

def main(argv):
    if len(argv) != 4:
        print(usage)
        return 1

    report = {}
    if argv[1] == "-d":
        argv[1] = "solution1/syn/report/csynth.xml"

    try:
        root = ET.parse(argv[1]).getroot()
    except OSError:
        print("Unable to read specified report file \"" + argv[1] + "\", aborting...")
        return 1

    print(root)

    user_assign = root.find('UserAssignments')
    perf_estim = root.find('PerformanceEstimates')
    area_estim = root.find('AreaEstimates/Resources')

    report['input'] = argv[2]
    report['optimizations'] = argv[3]

    report['part'] = user_assign.find('Part').text
    report['target_clock'] = user_assign.find('TargetClockPeriod').text

    report['estim_clock'] = perf_estim.find('SummaryOfTimingAnalysis/EstimatedClockPeriod').text
    report['lat_worst'] = perf_estim.find('SummaryOfOverallLatency/Worst-caseLatency').text
    report['lat_avg'] = perf_estim.find('SummaryOfOverallLatency/Average-caseLatency').text
    report['lat_best'] = perf_estim.find('SummaryOfOverallLatency/Best-caseLatency').text

    report['FF'] = area_estim.find('FF').text
    report['LUT'] = area_estim.find('LUT').text
    report['BRAM'] = area_estim.find('BRAM_18K').text
    report['DSP'] = area_estim.find('DSP48E').text

    fieldnames = report.keys()
    if path.exists('reports.csv'):
        print("reports.csv found in current directory, adding...")
        with open('reports.csv', 'a', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writerow(report)
    else:
        with open('reports.csv', 'w', newline='') as output:
            print("reports.csv not found, creating...")
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(report)
    print("Report for \"" + report['input'] + "\" successfully added to reports.csv")
    return 0

if __name__ == "__main__":
    main(sys.argv)