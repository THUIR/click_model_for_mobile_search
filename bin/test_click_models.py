#!/usr/bin/env pypy

import sys
import glob
import argparse
import numpy as np
import os

sys.path.append("..")
from clickmodels.inference import *
from clickmodels.input_reader import InputReader
from collections import Counter, defaultdict
import math
import json

try:
    from config import *
except:
    from clickmodels.config_sample import *


def DCG(rels, at=5):
    rels = [1.0 * r if r >= 0.0 else 0.0 for r in rels][0:at]
    rels = [2**r - 1.0 for r in rels]
    discount = [math.log(i+2, 2) for i in xrange(at)]
    ret = [r / d for r, d in zip(rels, discount)]
    for i in xrange(1, min(at, len(ret))):
        ret[i] += ret[i - 1]
    return ret

def read_df(filename, min_f=1, max_f=10000000):
    data = []
    for line in open(filename):
        e = line.split('\t')
        f = int(e[1])
        if f < min_f:
            continue
        if f > max_f:
            continue
        data.append([
                0,
                int(e[1]),
                float(e[2]),
                float(e[3])
        ] + json.loads(e[4]) + json.loads(e[5]))
    ret = np.array(data, dtype=np.float64)
    print >> sys.stderr, 'Test set size:', len(ret)
    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_dir', help="path to the training set")
    parser.add_argument('test_dir', help="path to the test set")
    parser.add_argument('-o', '--output', help="path to output directory")
    # parser.add_argument('-r', '--relevance_file',
    #                     help="if relevance file is given, the ndcg file for each test session will be computed")
    parser.add_argument('-m', '--model',
                        help='the name of click model [default=MCM]',
                        default='MCM')
    parser.add_argument('-N', '--num_train_files',
                        help='the first N training files will be used, 0 means use all the files [default=1]',
                        type=int,
                        default=1)
    parser.add_argument('-M', '--num_test_files',
                        help='the first M test files will be used, 0 means use all the files [default=1]',
                        type=int,
                        default=1)
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='increase output verbosity.')
    parser.add_argument('-i', '--ignore_no_clicks', action='store_true',
                        help='ignore sessions that have no clicks')
    parser.add_argument('-c', '--configs', help='additional configs')
    parser.add_argument('-t', '--viewport_time', action='store_true',
                        help='use viewport time data to train the model')
    parser.add_argument('-I', '--ignore_no_viewport', action='store_true',
                        help='ignore sessions with zero viewport time')
    parser.add_argument('-V', '--viewport_time_model', default=0,
                        type=int, help='choose viewport time model')
    parser.add_argument('-f', '--query_frequency', default=1,
                        type=int, help='the query frequency threshold for test')

    args = parser.parse_args()

    MODEL_CONSTRUCTORS = {
        'DBN': lambda config: DbnModel((0.9, 0.9, 0.9, 0.9), config=config),
        # 'DBN-layout': lambda config: DbnModel((1.0, 0.9, 1.0, 0.9), ignoreLayout=False, config=config),
        'UBM': lambda config: UbmModel(config=config),
        'DCM': lambda config: DcmModel(config=config),
        'UBM-layout': lambda config: UbmModel(ignoreLayout=False, ignoreVerticalType=False, config=config),
        'EBUBM': lambda config: UbmModel(explorationBias=True, config=config),
        'UBM-N': lambda config: McmModel(ignoreClickSatisfaction=True, ignoreExamSatisfaction=True, config=config),
        'UBM-CS': lambda config: McmModel(ignoreClickNecessity=True, ignoreExamSatisfaction=True, config=config),
        'MCM': lambda config: McmModel(config=config),
        'MCM-VPT': lambda config: McmVptModel(config=config, viewport_time_model=args.viewport_time_model),
        'MCM-VPT-OFF': lambda config: McmVptModel(config=config, useViewportTime=False),
    }

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    readInput = InputReader(MIN_DOCS_PER_QUERY, MAX_DOCS_PER_QUERY,
                            EXTENDED_LOG_FORMAT, SERP_SIZE,
                            TRAIN_FOR_METRIC,
                            discard_no_clicks=args.ignore_no_clicks,
                            read_viewport_time=args.viewport_time,
                            discard_no_viewport=args.ignore_no_viewport)

    print 'prepare training sessions...'
    train_files = sorted(glob.glob(args.train_dir + '/part-*'))
    if args.num_train_files > 0:
        train_files = train_files[0:args.num_train_files]

    train_sessions = []
    for fileNumber in xrange(len(train_files)):
        f = train_files[fileNumber]
        new_sessions = readInput(open(f))
        readInput.get_vertical_id_mapping(new_sessions)
        train_sessions += new_sessions
    print >> sys.stderr, 'Train set size:', len(train_sessions)
    max_train_query_id = readInput.current_query_id
    query_freq = Counter(s.query for s in train_sessions)

    print 'prepare test sessions...'
    test_files = sorted(glob.glob(args.test_dir + '/part-*'))
    if args.num_test_files > 0:
        test_files = test_files[0:args.num_train_files]
    for fileNumber in xrange(len(test_files)):
        f = test_files[fileNumber]
        new_sessions = readInput(open(f))
        readInput.get_vertical_id_mapping(new_sessions)

    if args.output is not None:
        with open(args.output + '/query_id.txt', 'w') as fout:
            for k, v in readInput.query_to_id.iteritems():
                query, region = k
                print >>fout, '%s\t%s\t%d\t%d' % (query, region, v, query_freq[v])
        with open(args.output + '/url_id.txt', 'w') as fout:
            for k, v in readInput.url_to_id.iteritems():
                print >>fout, ('%s\t%d' % (k, v)).encode('utf8', 'ignore')
        with open(args.output + '/query_url_vrid.txt', 'w') as fout:
            for k, v in readInput.qu_id_to_vrid.iteritems():
                qid, uid = k
                print >>fout, '%d\t%d\t%d' % (qid, uid, v)

    print 'train click model...'
    config = {
        'MAX_QUERY_ID': readInput.current_query_id + 1,
        'MAX_ITERATIONS': MAX_ITERATIONS,
        'DEBUG': DEBUG,
        'PRETTY_LOG': not args.verbose,
        'MAX_DOCS_PER_QUERY': MAX_DOCS_PER_QUERY,
        'SERP_SIZE': SERP_SIZE,
        'TRANSFORM_LOG': TRANSFORM_LOG,
        'QUERY_INDEPENDENT_PAGER': QUERY_INDEPENDENT_PAGER,
        'DEFAULT_REL': DEFAULT_REL,
        'MAX_VERTICAL_ID': readInput.max_vertical_id,
    }

    if args.configs:
        import json
        additional_configs = json.loads(args.configs)
        config.update(additional_configs)
        print config

    model_cls = MODEL_CONSTRUCTORS[args.model]
    m = model_cls(config)
    m.train(train_sessions)

    if args.output is not None:
        pred_fout = open(args.output + '/click_prediction.txt', 'w')

        for fileNumber in xrange(len(test_files)):
            f = test_files[fileNumber]
            for line in open(f):
                # get sess and line
                sess = readInput([line])
                if len(sess) == 0:
                    continue
                sess = sess[0]
                # skip the test session if the query was not seen in training set
                if sess.query >= max_train_query_id:
                    continue

                # click prediction
                entries = line.rstrip().split('\t')
                sid = entries[0]
                _ll, _perplexity, _positionPerplexity, _positionPerplexityClickSkip =\
                    m.test([sess], reportPositionPerplexity=True)
                _positionPerplexity = map(lambda x: -math.log(x, 2), _positionPerplexity)
                print >>pred_fout, '%s\t%d\t%f\t%f\t%s\t%s' % (
                    sid,
                    query_freq[sess.query],
                    _ll,
                    _perplexity,
                    str(_positionPerplexity),
                    str(sess.clicks)
                )

        pred_fout.close()

    data = []
    df = read_df(args.output + '/click_prediction.txt', min_f=args.query_frequency)
    data.append([
        np.mean(df[:, 2:3]),
        np.mean(np.mean(df[:, 4:9], axis=0)),
        np.mean(2 ** -np.mean(df[:, 4:14], axis=0)),
        np.mean(2 ** -np.mean(df[:, 4:9], axis=0)),
    ])
    print ', '.join(map(str, data[-1]))


    if args.output is not None:
        rel_fout = open(args.output + '/relevance_estimation.txt', 'w')

        for fileNumber in xrange(len(train_files)):
            f = train_files[fileNumber]
            for line in open(f):
                # get sess and line
                sess = readInput([line])
                if len(sess) == 0:
                    continue
                sess = sess[0]

                entries = line.rstrip().split('\t')
                sid = entries[0]

                alpha, beta, s_c, s_e = [], [], [], []
                for url, vrid in zip(sess.results, sess.layout):
                    _a, _b, _s_c, _s_e = m.get_relevance_parameters(sess.query, url, vrid)
                    alpha.append(_a)
                    beta.append(_b)
                    s_c.append(_s_c)
                    s_e.append(_s_e)

                query = entries[1]
                urls = json.loads(entries[4])

                print >>rel_fout, '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' % (
                    sid, query,
                    entries[4], # results
                    entries[5], # layout(vrids)
                    str(alpha),
                    str(beta),
                    str(s_c),
                    str(s_e)
                )
        rel_fout.close()
