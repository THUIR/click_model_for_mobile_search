#!/usr/bin/env pypy

import sys
import glob
import argparse
from collections import defaultdict
import heapq

try:
    import simplejson as json
except ImportError:
    import json

rel_funcs = {
    'DBN': lambda a, b, s_c, s_e: a * s_c,
    'UBM': lambda a, b, s_c, s_e: a,
    'DCM': lambda a, b, s_c, s_e: a,
    'UBM-layout': lambda a, b, s_c, s_e: a,
    'EBUBM': lambda a, b, s_c, s_e: a,
    'UBM-N': lambda a, b, s_c, s_e: a,
    'UBM-CS': lambda a, b, s_c, s_e: a,
    'MCM': lambda a, b, s_c, s_e: a * (b * s_c + (1.0 - b) * s_e),
    'MCM-VPT': lambda a, b, s_c, s_e: a * (b * s_c + (1.0 - b) * s_e),
}

def get_query_ranking(fin,
                      top_n=10,
                      rel_func=rel_funcs['MCM'],
                      filter_vertical=False):
    columns = ['query', 'rel', 'pred_rel', 'url', 'vrid', 'alpha', 'beta', 's_c', 's_e']
    data = defaultdict(dict)
    n = 0
    for line in fin:
        e = line.rstrip().split('\t')
        query = e[1]
        urls = json.loads(e[2])[0:10]
        vrids = json.loads(e[3])[0:10]
        rels = json.loads(e[4])[0:10]
        alphas = json.loads(e[5])[0:10]
        betas = json.loads(e[6])[0:10]
        s_c_list = json.loads(e[7])[0:10]
        s_e_list = json.loads(e[8])[0:10]

        for i in xrange(10):
            try:
                if filter_vertical:
                    if vrids[i] != '-1' and (not vrids[i].startswith('3')):
                        continue
                data[query][(urls[i], vrids[i])] = [rels[i],
                                                    rel_func(alphas[i], betas[i], s_c_list[i], s_e_list[i]),
                                                    alphas[i], betas[i], s_c_list[i], s_e_list[i]]
                # ranks[query][(urls[i], vrids[i])].append(1.0 * i)
            except IndexError:
                break
        n += 1
        if n % 100000 == 0:
            print 'processed %d sessions...' % n

    print '#unique queries: %d' % len(data)

    ret = []
    for query in data:
        if top_n == -1:
            results = sorted(data[query].items(), key=lambda x: -x[1][1])
        else:
            results = heapq.nlargest(top_n, data[query].iteritems(), key=lambda x: x[1][1])

        for k, v in results:
            ret.append([query] + v[0:2] + list(k) + v[2:])

    return ret

def to_str(x):
    if isinstance(x, unicode):
        x = x.encode('utf8')
    return str(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help="path to the input file")
    parser.add_argument('output', help="path to output file")
    parser.add_argument('-n', '--top_n',
                        help='only return the top n results, -1 means return all results',
                        type=int,
                        default=10)
    parser.add_argument('-m', '--model',
                        help='the name of click model [default=MCM]',
                        default='MCM')
    parser.add_argument('-f', '--filter_vertical', action='store_true',
                        help='filter the vertical results')
    args = parser.parse_args()

    with open(args.input, 'r') as fin:
        ret = get_query_ranking(fin,
                                top_n=args.top_n,
                                rel_func=rel_funcs[args.model],
                                filter_vertical=args.filter_vertical)
    with open(args.output, 'w') as fout:
        for row in ret:
            print >>fout, '\t'.join(map(to_str, row))


