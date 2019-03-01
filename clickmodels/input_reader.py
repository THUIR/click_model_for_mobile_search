from collections import namedtuple, defaultdict
import json

DEBUG = False

SessionItem = namedtuple('SessionItem', ['intentWeight', 'query', 'results', 'layout', 'clicks', 'extraclicks'])

class InputReader:
    def __init__(self, min_docs_per_query, max_docs_per_query,
                 extended_log_format, serp_size,
                 train_for_metric,
                 discard_no_clicks=False,
                 read_viewport_time=False,
                 discard_no_viewport=False):
        self.url_to_id = {}
        self.query_to_id = {}
        self.vrid_to_vertical_id = {}
        self.qu_id_to_vrid = defaultdict(lambda: 0)
        self.current_url_id = 1
        self.current_query_id = 0
        self.current_vertical_id = 0
        self.max_vertical_id = 0

        self.min_docs_per_query = min_docs_per_query
        self.max_docs_per_query = max_docs_per_query
        self.extended_log_format = extended_log_format
        self.serp_size = serp_size
        self.train_for_metric = train_for_metric
        self.discard_no_clicks = discard_no_clicks
        self.read_viewport_time = read_viewport_time
        self.discard_no_viewport = discard_no_viewport

    def __call__(self, f):
        sessions = []
        for line in f:
            entries = line.rstrip().split('\t')
            hash_digest, query, region, intentWeight, urls, layout, clicks = entries[0:7]
            urls, layout, clicks = map(json.loads, [urls, layout, clicks])
            extra = {}
            if self.read_viewport_time:
                vpt, w_vpt, height = entries[7:10]
                vpt, w_vpt, height = map(json.loads, [vpt, w_vpt, height])
                extra.update(zip(['vpt', 'w_vpt', 'height'], [vpt, w_vpt, height]))
                if self.discard_no_viewport and sum(vpt) <= 0:
                    continue
            urlsObserved = 0
            if self.extended_log_format:
                maxLen = self.max_docs_per_query
                if TRANSFORM_LOG:
                    maxLen -= self.max_docs_per_query // self.serp_size
                urls, _ = self.convertToList(urls, '', maxLen)
                for u in urls:
                    if u == '':
                        break
                    urlsObserved += 1
                urls = urls[:urlsObserved]
                layout, _ = self.convertToList(layout, False, urlsObserved)
                clicks, extra = self.convertToList(clicks, 0, urlsObserved)
            else:
                urls = urls[:self.max_docs_per_query]
                urlsObserved = len(urls)
                layout = layout[:urlsObserved]
                clicks = clicks[:urlsObserved]
            if urlsObserved < self.min_docs_per_query:
                continue
            if self.discard_no_clicks and not any(clicks):
                continue
            if float(intentWeight) > 1 or float(intentWeight) < 0:
                continue
            if (query, region) in self.query_to_id:
                query_id = self.query_to_id[(query, region)]
            else:
                query_id = self.current_query_id
                self.query_to_id[(query, region)] = self.current_query_id
                self.current_query_id += 1
            intentWeight = float(intentWeight)

            for idx, vr_id in enumerate(layout):
                if vr_id not in self.vrid_to_vertical_id:
                    self.vrid_to_vertical_id[vr_id] = self.current_vertical_id
                    self.current_vertical_id += 1
                layout[idx] = self.vrid_to_vertical_id[vr_id]
            # add fake G_{self.max_docs_per_query+1} to simplify gamma calculation:
            # layout.append(False)
            layout.append(0)
            url_ids = []
            for u in urls:
                if u in ['_404', 'STUPID', 'VIRUS', 'SPAM']:
                    # convert Yandex-specific fields to standard ones
                    assert self.train_for_metric
                    u = 'IRRELEVANT'
                if u.startswith('RELEVANT_'):
                    # convert Yandex-specific fields to standard ones
                    assert self.train_for_metric
                    u = 'RELEVANT'
                if u in self.url_to_id:
                    if self.train_for_metric:
                        url_ids.append(u)
                    else:
                        url_ids.append(self.url_to_id[u])
                else:
                    urlid = self.current_url_id
                    if self.train_for_metric:
                        url_ids.append(u)
                    else:
                        url_ids.append(urlid)
                    self.url_to_id[u] = urlid
                    self.current_url_id += 1
            sessions.append(SessionItem(intentWeight, query_id, url_ids, layout, clicks, extra))
        self.max_vertical_id = self.current_vertical_id
        return sessions

    def get_vertical_id_mapping(self, sessions):
        for sess in sessions:
            for url_id, vrid in zip(sess.results, sess.layout):
                self.qu_id_to_vrid[(sess.query, url_id)] = vrid

    @staticmethod
    def convertToList(sparseDict, defaultElem, maxLen):
        """ Convert dict of the format {"0": doc0, "13": doc13} to the list of the length maxLen"""
        convertedList = [defaultElem] * maxLen
        extra = {}
        for k, v in sparseDict.iteritems():
            try:
                convertedList[int(k)] = v
            except (ValueError, IndexError):
                extra[k] = v
        return convertedList, extra

    @staticmethod
    def mergeExtraToSessionItem(s, serp_size):
        """ Put pager click into the session item (presented as a fake URL) """
        if s.extraclicks.get('TRANSFORMED', False):
            return s
        else:
            newUrls = []
            newLayout = []
            newClicks = []
            a = 0
            while a + serp_size <= len(s.results):
                b = a + serp_size
                newUrls += s.results[a:b]
                newLayout += s.layout[a:b]
                newClicks += s.clicks[a:b]
                # TODO: try different fake urls for different result pages (page_1, page_2, etc.)
                newUrls.append('PAGER')
                newLayout.append(False)
                newClicks.append(1)
                a = b
            newClicks[-1] = 0 if a == len(s.results) else 1
            newLayout.append(False)
            if DEBUG:
                assert len(newUrls) == len(newClicks)
                assert len(newUrls) + 1 == len(newLayout), (len(newUrls), len(newLayout))
                assert len(newUrls) < len(s.results) + self.max_docs_per_query / serp_size, (len(s.results), len(newUrls))
            return SessionItem(s.intentWeight, s.query, newUrls, newLayout, newClicks, {'TRANSFORMED': True})

    def read_rel_file(self, f):
        """read file that consists of such lines:
            query\t[url | ...]\trel

            return list of (line, query_id, url_id, vertical_id)
        """
        ret = []
        for line in f:
            line = line.rstrip()
            query, urls, _ = line.split('\t', 2)
            query = (query, '0')
            url = urls.split('|')[0].strip()
            url = url.decode('utf8')
            if query in self.query_to_id and url in self.url_to_id:
                qid, uid = self.query_to_id[query], self.url_to_id[url]
                vertical_id = self.qu_id_to_vrid[(qid, uid)]
                ret.append((line, qid, uid, vertical_id))
        return ret




