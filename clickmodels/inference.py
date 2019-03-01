import random
import sys
from collections import defaultdict
from datetime import datetime

from viewport_time_model import *


# import scipy
# from scipy.special import psi
# MAX_ITERATIONS, DEBUG, PRETTY_LOG, MAX_DOCS_PER_QUERY, SERP_SIZE, \
# TRANSFORM_LOG, QUERY_INDEPENDENT_PAGER, DEFAULT_REL, MAX_VERTICAL_ID, \
# DEFAULT_SAT_CLICK, DEFAULT_SAT_EXAM, \
# ALPHA_PRIOR, BETA_PRIOR, S_C_PRIOR, S_E_PRIOR, VPT_EPSILON

class NotImplementedError(Exception):
    pass


class ClickModel:
    """
        An abstract click model interface.
    """

    def __init__(self, ignoreIntents=True, ignoreLayout=True, config=None):
        self.config = config if config is not None else {}
        self.ignoreIntents = ignoreIntents
        self.ignoreLayout = ignoreLayout

    def train(self, sessions):
        """
            Trains the model.
        """
        pass

    def test(self, sessions, reportPositionPerplexity=True):
        """
            Evaluates the prediciton power of the click model for a given sessions.
            Returns the log-likelihood, perplexity, position perplexity
            (perplexity for each rank a.k.a. position in a SERP)
            and separate perplexity values for clicks and non-clicks (skips).
        """
        logLikelihood = 0.0
        positionPerplexity = [0.0] * self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY)
        positionPerplexityClickSkip = [[0.0, 0.0] \
                                       for i in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))]
        counts = [0] * self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY)
        countsClickSkip = [[0, 0] \
                           for i in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))]
        possibleIntents = [False] if self.ignoreIntents else [False, True]
        for s in sessions:
            iw = s.intentWeight
            intentWeight = {False: 1.0} if self.ignoreIntents else {False: 1 - iw, True: iw}
            clickProbs = self._get_click_probs(s, possibleIntents)
            N = min(len(s.clicks), self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))
            if self.config.get('DEBUG', DEBUG):
                assert N > 1
                x = sum(clickProbs[i][N // 2] * intentWeight[i] for i in possibleIntents) / sum(
                    clickProbs[i][N // 2 - 1] * intentWeight[i] for i in possibleIntents)
                s.clicks[N // 2] = 1 if s.clicks[N // 2] == 0 else 0
                clickProbs2 = self._get_click_probs(s, possibleIntents)
                y = sum(clickProbs2[i][N // 2] * intentWeight[i] for i in possibleIntents) / sum(
                    clickProbs2[i][N // 2 - 1] * intentWeight[i] for i in possibleIntents)
                assert abs(x + y - 1) < 0.01, (x, y)
            # Marginalize over possible intents: P(C_1, ..., C_N) = \sum_{i} P(C_1, ..., C_N | I=i) P(I=i)
            logLikelihood += math.log(sum(clickProbs[i][N - 1] * intentWeight[i] for i in possibleIntents)) / N
            correctedRank = 0  # we are going to skip clicks on fake pager urls
            for k, click in enumerate(s.clicks):
                click = 1 if click else 0
                if s.extraclicks.get('TRANSFORMED', False) and \
                                        (k + 1) % (self.config.get('SERP_SIZE', SERP_SIZE) + 1) == 0:
                    if self.config.get('DEBUG', DEBUG):
                        assert s.results[k] == 'PAGER'
                    continue
                # P(C_k | C_1, ..., C_{k-1}) = \sum_I P(C_1, ..., C_k | I) P(I) / \sum_I P(C_1, ..., C_{k-1} | I) P(I)
                curClick = dict((i, clickProbs[i][k]) for i in possibleIntents)
                prevClick = dict((i, clickProbs[i][k - 1]) for i in possibleIntents) if k > 0 else dict(
                    (i, 1.0) for i in possibleIntents)
                logProb = math.log(sum(curClick[i] * intentWeight[i] for i in possibleIntents), 2) - math.log(
                    sum(prevClick[i] * intentWeight[i] for i in possibleIntents), 2)
                positionPerplexity[correctedRank] += logProb
                positionPerplexityClickSkip[correctedRank][click] += logProb
                counts[correctedRank] += 1
                countsClickSkip[correctedRank][click] += 1
                correctedRank += 1
        positionPerplexity = [2 ** (-x / count if count else x) for (x, count) in zip(positionPerplexity, counts)]
        positionPerplexityClickSkip = [[2 ** (-x[click] / (count[click] if count[click] else 1) if count else x) \
                                        for (x, count) in zip(positionPerplexityClickSkip, countsClickSkip)] for click
                                       in xrange(2)]
        perplexity = sum(positionPerplexity) / len(positionPerplexity)
        if reportPositionPerplexity:
            return logLikelihood / len(sessions), perplexity, positionPerplexity, positionPerplexityClickSkip
        else:
            return logLikelihood / len(sessions), perplexity

    def _get_click_probs(self, s, possible_intents):
        """
            Returns click probabilities list for a given list of s.clicks.
            For each intent $i$ and each rank $k$ we have:
            click_probs[i][k-1] = P(C_1, ..., C_k | I=i)
        """
        click_probs = dict((i, [0.5 ** (k + 1) for k in xrange(len(s.clicks))]) for i in possible_intents)
        return click_probs

    def get_loglikelihood(self, sessions):
        """
            Returns the average log-likelihood of the current model for given sessions.
            This is a lightweight version of the self.test() method.
        """
        return sum(self.get_log_click_probs(s) for s in sessions) / len(sessions)

    def get_log_click_probs(self, session):
        """
            Returns an average log-likelihood for a given session,
            i.e. log-likelihood of all the click events, divided
            by the number of documents in the session.
        """
        possibleIntents = [False] if self.ignoreIntents else [False, True]
        intentWeight = {False: 1.0} if self.ignoreIntents else \
            {False: 1 - session.intentWeight, True: session.intentWeight}
        clickProbs = self._get_click_probs(s, possibleIntents)
        N = min(len(session.clicks), self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))
        # Marginalize over possible intents: P(C_1, ..., C_N) = \sum_{i} P(C_1, ..., C_N | I=i) P(I=i)
        return math.log(sum(clickProbs[i][N - 1] * intentWeight[i] for i in possibleIntents)) / N

    def get_model_relevances(self, session, intent=False):
        """
            Returns estimated relevance of each document in a given session
            based on a trained click model.
        """
        raise NotImplementedError

    def predict_click_probs(self, session, intent=False):
        """
            Predicts click probabilities for a given session. Does not use session.clicks.
            This is a vector of P(C_k = 1 | E_k = 1) for different ranks $k$.
        """
        raise NotImplementedError

    def predict_stop_probs(self, session, intent=False):
        """
            Predicts stop probabilities (after click) for each document in a session.
            This is often referred to as satisfaction probability.
            This is a vector of P(S_k = 1 | C_k = 1) for different ranks $k$.
        """
        raise NotImplementedError

    def get_abandonment_prob(self, rank, intent=False, layout=None):
        """
            Predicts probability of stopping without click after examining document at rank `rank`.
        """
        return 0.0

    def generate_clicks(self, session):
        """
            Generates clicks for a given session, assuming cascade examination order.
        """
        clicks = [0] * len(session.results)
        # First, randomly select user intent.
        intent = False  # non-vertical intent by default
        if not self.ignoreIntents:
            random_intent_prob = random.uniforma(0, 1)
            if random_intent_prob < session.intentWeight:
                intent = True
        predicted_click_probs = self.predict_click_probs(session, intent)
        predicted_stop_probs = self.predict_stop_probs(session, intent)
        for rank, result in enumerate(session.results):
            random_click_prob = random.uniform(0, 1)
            clicks[rank] = 1 if random_click_prob < predicted_click_probs[rank] else 0
            if clicks[rank] == 1:
                random_stop_prob = random.uniform(0, 1)
                if random_stop_prob < predicted_stop_probs[rank]:
                    break
            else:
                random_stop_prob = random.uniform(0, 1)
                if random_stop_prob < self.get_abandonment_prob(rank, intent, session.layout):
                    break
        return clicks


class DbnModel(ClickModel):
    def __init__(self, gammas, ignoreIntents=True, ignoreLayout=True, config=None):
        self.gammas = gammas
        ClickModel.__init__(self, ignoreIntents, ignoreLayout, config)

    def train(self, sessions):
        possibleIntents = [False] if self.ignoreIntents else [False, True]
        max_query_id = self.config.get('MAX_QUERY_ID')
        if max_query_id is None:
            print >> sys.stderr, 'WARNING: no MAX_QUERY_ID specified for', self
            max_query_id = 100000
        # intent -> query -> url -> (a_u, s_u)
        self.urlRelevances = dict((i,
                                   [defaultdict(lambda: {'a': self.config.get('DEFAULT_REL', DEFAULT_REL),
                                                         's': self.config.get('DEFAULT_REL', DEFAULT_REL)}) \
                                    for q in xrange(max_query_id)]) for i in possibleIntents
                                  )
        # here we store distribution of posterior intent weights given train data
        self.queryIntentsWeights = defaultdict(lambda: [])
        alpha_priors = self.config.get('ALPHA_PRIOR', ALPHA_PRIOR)
        alpha_priors[1] -= alpha_priors[0]
        s_c_priors = self.config.get('S_C_PRIOR', S_C_PRIOR)
        s_c_priors[1] -= s_c_priors[0]

        # EM algorithm
        if not self.config.get('PRETTY_LOG', PRETTY_LOG):
            print >> sys.stderr, '-' * 80
            print >> sys.stderr, 'Start. Current time is', datetime.now()

        for iteration_count in xrange(self.config.get('MAX_ITERATIONS', MAX_ITERATIONS)):
            # urlRelFractions[intent][query][url][r][1] --- coefficient before \log r
            # urlRelFractions[intent][query][url][r][0] --- coefficient before \log (1 - r)

            urlRelFractions = dict((i, [defaultdict(lambda: {'a': list(alpha_priors), 's': list(s_c_priors)})
                                        for q in xrange(max_query_id)]) for i in [False, True])
            self.queryIntentsWeights = defaultdict(lambda: [])
            # E step
            for s in sessions:
                positionRelevances = {}
                query = s.query
                for intent in possibleIntents:
                    positionRelevances[intent] = {}
                    for r in ['a', 's']:
                        positionRelevances[intent][r] = [self.urlRelevances[intent][query][url][r] for url in s.results]
                layout = [False] * len(s.layout) if self.ignoreLayout else s.layout
                sessionEstimate = dict(
                    (intent, self._getSessionEstimate(positionRelevances[intent], layout, s.clicks, intent)) for intent
                    in possibleIntents)
                # P(I | C, G)
                if self.ignoreIntents:
                    p_I__C_G = {False: 1, True: 0}
                else:
                    a = sessionEstimate[False]['C'] * (1 - s.intentWeight)
                    b = sessionEstimate[True]['C'] * s.intentWeight
                    p_I__C_G = {False: a / (a + b), True: b / (a + b)}
                self.queryIntentsWeights[query].append(p_I__C_G[True])
                for k, url in enumerate(s.results):
                    for intent in possibleIntents:
                        # update a
                        urlRelFractions[intent][query][url]['a'][1] += sessionEstimate[intent]['a'][k] * p_I__C_G[
                            intent]
                        urlRelFractions[intent][query][url]['a'][0] += (1 - sessionEstimate[intent]['a'][k]) * p_I__C_G[
                            intent]
                        if s.clicks[k] != 0:
                            # Update s
                            urlRelFractions[intent][query][url]['s'][1] += sessionEstimate[intent]['s'][k] * p_I__C_G[
                                intent]
                            urlRelFractions[intent][query][url]['s'][0] += (1 - sessionEstimate[intent]['s'][k]) * \
                                                                           p_I__C_G[intent]
            if not self.config.get('PRETTY_LOG', PRETTY_LOG):
                sys.stderr.write('E')
            # M step
            # update parameters and record mean square error
            sum_square_displacement = 0.0
            Q_functional = 0.0
            for i in possibleIntents:
                for query, d in enumerate(urlRelFractions[i]):
                    if not d:
                        continue
                    for url, relFractions in d.iteritems():
                        a_u_new = relFractions['a'][1] / (relFractions['a'][1] + relFractions['a'][0])
                        sum_square_displacement += (a_u_new - self.urlRelevances[i][query][url]['a']) ** 2
                        self.urlRelevances[i][query][url]['a'] = a_u_new
                        Q_functional += relFractions['a'][1] * math.log(a_u_new) + relFractions['a'][0] * math.log(
                            1 - a_u_new)
                        s_u_new = relFractions['s'][1] / (relFractions['s'][1] + relFractions['s'][0])
                        sum_square_displacement += (s_u_new - self.urlRelevances[i][query][url]['s']) ** 2
                        self.urlRelevances[i][query][url]['s'] = s_u_new
                        Q_functional += relFractions['s'][1] * math.log(s_u_new) + relFractions['s'][0] * math.log(
                            1 - s_u_new)
            if not self.config.get('PRETTY_LOG', PRETTY_LOG):
                sys.stderr.write('M\n')
            rmsd = math.sqrt(sum_square_displacement)
            if self.config.get('PRETTY_LOG', PRETTY_LOG):
                sys.stderr.write('%d..' % (iteration_count + 1))
            else:
                print >> sys.stderr, 'Iteration: %d, ERROR: %f' % (iteration_count + 1, rmsd)
                print >> sys.stderr, 'Q functional: %f' % Q_functional
        if self.config.get('PRETTY_LOG', PRETTY_LOG):
            sys.stderr.write('\n')
        for q, intentWeights in self.queryIntentsWeights.iteritems():
            self.queryIntentsWeights[q] = sum(intentWeights) / len(intentWeights)

    @staticmethod
    def testBackwardForward():
        positionRelevances = {'a': [0.5] * MAX_DOCS_PER_QUERY, 's': [0.5] * MAX_DOCS_PER_QUERY}
        gammas = [0.9] * 4
        layout = [False] * (MAX_DOCS_PER_QUERY + 1)
        clicks = [0] * MAX_DOCS_PER_QUERY
        alpha, beta = DbnModel.getForwardBackwardEstimates(positionRelevances, gammas, layout, clicks, False)
        x = alpha[0][0] * beta[0][0] + alpha[0][1] * beta[0][1]
        assert all(abs((a[0] * b[0] + a[1] * b[1]) / x - 1) < 0.00001 for a, b in zip(alpha, beta))

    @staticmethod
    def getGamma(gammas, k, layout, intent):
        index = 2 * (1 if layout[k + 1] else 0) + (1 if intent else 0)
        return gammas[index]

    @staticmethod
    def getForwardBackwardEstimates(positionRelevances, gammas, layout, clicks, intent,
                                    debug=False):
        N = len(clicks)
        if debug:
            assert N + 1 == len(layout)
        alpha = [[0.0, 0.0] for i in xrange(N + 1)]
        beta = [[0.0, 0.0] for i in xrange(N + 1)]
        alpha[0] = [0.0, 1.0]
        beta[N] = [1.0, 1.0]
        # P(E_{k+1} = e, C_k | E_k = e', G, I)
        updateMatrix = [[[0.0 for e1 in [0, 1]] for e in [0, 1]] for i in xrange(N)]
        for k, C_k in enumerate(clicks):
            a_u = positionRelevances['a'][k]
            s_u = positionRelevances['s'][k]
            gamma = DbnModel.getGamma(gammas, k, layout, intent)
            if C_k == 0:
                updateMatrix[k][0][0] = 1
                updateMatrix[k][0][1] = (1 - gamma) * (1 - a_u)
                updateMatrix[k][1][0] = 0
                updateMatrix[k][1][1] = gamma * (1 - a_u)
            else:
                updateMatrix[k][0][0] = 0
                updateMatrix[k][0][1] = (s_u + (1 - gamma) * (1 - s_u)) * a_u
                updateMatrix[k][1][0] = 0
                updateMatrix[k][1][1] = gamma * (1 - s_u) * a_u

        for k in xrange(N):
            for e in [0, 1]:
                alpha[k + 1][e] = sum(alpha[k][e1] * updateMatrix[k][e][e1] for e1 in [0, 1])
                beta[N - 1 - k][e] = sum(beta[N - k][e1] * updateMatrix[N - 1 - k][e1][e] for e1 in [0, 1])
        return alpha, beta

    def _getSessionEstimate(self, positionRelevances, layout, clicks, intent):
        # Returns {'a': P(A_k | I, C, G), 's': P(S_k | I, C, G), 'C': P(C | I, G), 'clicks': P(C_k | C_1, ..., C_{k-1}, I, G)} as a dict
        # sessionEstimate[True]['a'][k] = P(A_k = 1 | I = 'Fresh', C, G), probability of A_k = 0 can be calculated as 1 - p
        N = len(clicks)
        if self.config.get('DEBUG', DEBUG):
            assert N + 1 == len(layout)
        sessionEstimate = {'a': [0.0] * N, 's': [0.0] * N, 'e': [[0.0, 0.0] for k in xrange(N)], 'C': 0.0,
                           'clicks': [0.0] * N}
        alpha, beta = self.getForwardBackwardEstimates(positionRelevances,
                                                       self.gammas, layout, clicks, intent,
                                                       debug=self.config.get('DEBUG', DEBUG)
                                                       )
        try:
            varphi = [((a[0] * b[0]) / (a[0] * b[0] + a[1] * b[1]), (a[1] * b[1]) / (a[0] * b[0] + a[1] * b[1])) for
                      a, b in zip(alpha, beta)]
        except ZeroDivisionError:
            print >> sys.stderr, alpha, beta, [(a[0] * b[0] + a[1] * b[1]) for a, b in
                                               zip(alpha, beta)], positionRelevances
            sys.exit(1)
        if self.config.get('DEBUG', DEBUG):
            assert all(ph[0] < 0.01 for ph, c in zip(varphi[:N], clicks) if c != 0), (alpha, beta, varphi, clicks)
        # calculate P(C | I, G) for k = 0
        sessionEstimate['C'] = alpha[0][0] * beta[0][0] + alpha[0][1] * beta[0][1]  # == 0 + 1 * beta[0][1]
        for k, C_k in enumerate(clicks):
            a_u = positionRelevances['a'][k]
            s_u = positionRelevances['s'][k]
            gamma = self.getGamma(self.gammas, k, layout, intent)
            # E_k_multiplier --- P(S_k = 0 | C_k) P(C_k | E_k = 1)
            if C_k == 0:
                sessionEstimate['a'][k] = a_u * varphi[k][0]
                sessionEstimate['s'][k] = 0.0
            else:
                sessionEstimate['a'][k] = 1.0
                sessionEstimate['s'][k] = varphi[k + 1][0] * s_u / (s_u + (1 - gamma) * (1 - s_u))
            # P(C_1, ..., C_k | I)
            sessionEstimate['clicks'][k] = sum(alpha[k + 1])
        return sessionEstimate

    def _get_click_probs(self, s, possibleIntents):
        """
            Returns clickProbs list:
            clickProbs[i][k] = P(C_1, ..., C_k | I=i)
            """
        # TODO: ensure that s.clicks[l] not used to calculate clickProbs[i][k] for l >= k
        positionRelevances = {}
        for intent in possibleIntents:
            positionRelevances[intent] = {}
            for r in ['a', 's']:
                positionRelevances[intent][r] = [self.urlRelevances[intent][s.query][url][r] for url in s.results]
                if self.config.get('QUERY_INDEPENDENT_PAGER', QUERY_INDEPENDENT_PAGER):
                    for k, u in enumerate(s.results):
                        if u == 'PAGER':
                            # use dummy 0 query for all fake pager URLs
                            positionRelevances[intent][r][k] = self.urlRelevances[intent][0][url][r]
        layout = [False] * len(s.layout) if self.ignoreLayout else s.layout
        return dict((i, self._getSessionEstimate(positionRelevances[i], layout, s.clicks, i)['clicks']) for i in
                    possibleIntents)

    def get_model_relevances(self, session, intent=False):
        """
            Returns estimated relevance of each document in a given session
            based on a trained click model.

            You can make use of the fact that model trains different relevances
            for different intents by specifying `intent` argument. If it is set
            to False, simple web relevance is returned, if it is to True, then
            vertical relevance is returned, i.e., how relevant each document
            is to a vertical intent.
        """
        relevances = []
        for rank, result in enumerate(session.results):
            a = self.urlRelevances[intent][session.query][result]['a']
            s = self.urlRelevances[intent][session.query][result]['s']
            relevances.append(a * s)
        return relevances

    def get_relevance_parameters(self, query, url, vertical_id):
        """return alpha, beta, s_c, s_e"""
        a = self.urlRelevances[False][query][url]['a']
        s = self.urlRelevances[False][query][url]['s']
        return a, 0.0, s, 0.0

    def predict_click_probs(self, session, intent=False):
        """
            Predicts click probabilities for a given session. Does not use clicks.
        """
        click_probs = []
        for rank, result in enumerate(session.results):
            a = self.urlRelevances[intent][session.query][result]['a']
            click_probs.append(a)
        return click_probs

    def predict_stop_probs(self, session, intent=False):
        """
            Predicts stop probabilities for each document in a session.
        """
        stop_probs = []
        for rank, result in enumerate(session.results):
            s = self.urlRelevances[intent][session.query][result]['s']
            stop_probs.append(s)
        return stop_probs

    def get_abandonment_prob(self, rank, intent=False, layout=None):
        """
            Predicts probability of stopping without click after examining document at rank `rank`.
        """
        return 1.0 - self.getGamma(self.gammas, rank, layout, intent)


class SimplifiedDbnModel(DbnModel):
    def __init__(self, ignoreIntents=True, ignoreLayout=True, config=None):
        assert ignoreIntents
        assert ignoreLayout
        DbnModel.__init__(self, (1.0, 1.0, 1.0, 1.0), ignoreIntents, ignoreLayout, config)

    def train(self, sessions):
        max_query_id = self.config.get('MAX_QUERY_ID')
        if max_query_id is None:
            print >> sys.stderr, 'WARNING: no MAX_QUERY_ID specified for', self
            max_query_id = 100000
        urlRelFractions = [defaultdict(lambda: {'a': [1.0, 1.0], 's': [1.0, 1.0]}) for q in xrange(max_query_id)]
        for s in sessions:
            query = s.query
            lastClickedPos = len(s.clicks) - 1
            for k, c in enumerate(s.clicks):
                if c != 0:
                    lastClickedPos = k
            for k, (u, c) in enumerate(zip(s.results, s.clicks[:(lastClickedPos + 1)])):
                tmpQuery = query
                if self.config.get('QUERY_INDEPENDENT_PAGER', QUERY_INDEPENDENT_PAGER) \
                        and u == 'PAGER':
                    assert self.config.get('TRANSFORM_LOG', TRANSFORM_LOG)
                    # the same dummy query for all pagers
                    query = 0
                if c != 0:
                    urlRelFractions[query][u]['a'][1] += 1
                    if k == lastClickedPos:
                        urlRelFractions[query][u]['s'][1] += 1
                    else:
                        urlRelFractions[query][u]['s'][0] += 1
                else:
                    urlRelFractions[query][u]['a'][0] += 1
                if self.config.get('QUERY_INDEPENDENT_PAGER', QUERY_INDEPENDENT_PAGER):
                    query = tmpQuery
        self.urlRelevances = dict((i,
                                   [defaultdict(lambda: {'a': self.config.get('DEFAULT_REL', DEFAULT_REL),
                                                         's': self.config.get('DEFAULT_REL', DEFAULT_REL)}) \
                                    for q in xrange(max_query_id)]) for i in [False])
        for query, d in enumerate(urlRelFractions):
            if not d:
                continue
            for url, relFractions in d.iteritems():
                self.urlRelevances[False][query][url]['a'] = relFractions['a'][1] / (
                relFractions['a'][1] + relFractions['a'][0])
                self.urlRelevances[False][query][url]['s'] = relFractions['s'][1] / (
                relFractions['s'][1] + relFractions['s'][0])


class UbmModel(ClickModel):
    def __init__(self, ignoreIntents=True, ignoreLayout=True, explorationBias=False,
                 ignoreVerticalType=True, ignoreClickNecessity=True,
                 config=None):
        self.explorationBias = explorationBias
        self.ignoreVerticalType = ignoreVerticalType
        self.ignoreClickNecessity = ignoreClickNecessity

        self.gammaTypesNum = 4

        ClickModel.__init__(self, ignoreIntents, ignoreLayout, config)
        print >> sys.stderr, 'UbmModel:' + \
                             ' ignoreIntents=' + str(self.ignoreIntents) + \
                             ' ignoreLayout=' + str(self.ignoreLayout) + \
                             ' explorationBias=' + str(self.explorationBias) + \
                             ' ignoreVerticalType=' + str(self.ignoreVerticalType) + \
                             ' ignoreClickNecessity=' + str(self.ignoreClickNecessity)

        if self.ignoreClickNecessity and not self.ignoreVerticalType:
            self.gammaTypesNum = (self.config.get('MAX_VERTICAL_ID', MAX_VERTICAL_ID) + 1) * 2
            self.getGamma = self.getGammaWithVerticalId

    def train(self, sessions):
        max_query_id = self.config.get('MAX_QUERY_ID')
        if max_query_id is None:
            print >> sys.stderr, 'WARNING: no MAX_QUERY_ID specified for', self
            max_query_id = 100000
        possibleIntents = [False] if self.ignoreIntents else [False, True]
        # alpha: intent -> query -> url -> "attractiveness probability"
        self.alpha = dict((i,
                           [defaultdict(lambda: self.config.get('DEFAULT_REL', DEFAULT_REL)) \
                            for q in xrange(max_query_id)]) for i in possibleIntents)
        # gamma: freshness of the current result: gammaType -> rank -> "distance from the last click" - 1 -> examination probability
        self.gamma = [[[0.5 \
                        for d in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))] \
                       for r in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))] \
                      for g in xrange(self.gammaTypesNum)]
        if self.explorationBias:
            self.e = [0.5 \
                      for p in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))]
        if not self.ignoreClickNecessity:
            # vertical_id = 0 => organic docs
            self.cn = [0.5 \
                       for vertical_id in xrange(self.config.get('MAX_VERTICAL_ID', MAX_VERTICAL_ID) + 1)]
        if not self.config.get('PRETTY_LOG', PRETTY_LOG):
            print >> sys.stderr, '-' * 80
            print >> sys.stderr, 'Start. Current time is', datetime.now()
        for iteration_count in xrange(self.config.get('MAX_ITERATIONS', MAX_ITERATIONS)):
            self.queryIntentsWeights = defaultdict(lambda: [])
            # not like in DBN! xxxFractions[0] is a numerator while xxxFraction[1] is a denominator
            alphaFractions = dict((i, [defaultdict(lambda: list(self.config.get('ALPHA_PRIOR', ALPHA_PRIOR)))
                                       for q in xrange(max_query_id)]) for i in possibleIntents)
            gammaFractions = [[[[1.0, 2.0] \
                                for d in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))] \
                               for r in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))] \
                              for g in xrange(self.gammaTypesNum)]
            if self.explorationBias:
                eFractions = [[1.0, 2.0] \
                              for p in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))]
            if not self.ignoreClickNecessity:
                cnFractions = [[1.0, 2.0] \
                               for vertical_id in xrange(self.config.get('MAX_VERTICAL_ID', MAX_VERTICAL_ID) + 1)]
            # E-step
            for s in sessions:
                query = s.query
                layout = [0] * len(s.layout) if self.ignoreLayout else s.layout
                if self.explorationBias:
                    explorationBiasPossible = any((l and c for (l, c) in zip(s.layout, s.clicks)))
                    firstVerticalPos = -1 if not any(s.layout[:-1]) else [k for (k, l) in enumerate(s.layout) if l][0]
                if self.ignoreIntents:
                    p_I__C_G = {False: 1.0, True: 0}
                else:
                    a = self._getSessionProb(s) * (1 - s.intentWeight)
                    b = 1 * s.intentWeight
                    p_I__C_G = {False: a / (a + b), True: b / (a + b)}
                self.queryIntentsWeights[query].append(p_I__C_G[True])
                prevClick = -1
                for rank, c in enumerate(s.clicks):
                    url = s.results[rank]
                    for intent in possibleIntents:
                        a = self.alpha[intent][query][url]
                        if self.explorationBias and explorationBiasPossible:
                            e = self.e[firstVerticalPos]
                        if c == 0:
                            g = self.getGamma(self.gamma, rank, prevClick, layout, intent)
                            gCorrection = 1
                            if self.explorationBias and explorationBiasPossible and not s.layout[k]:
                                gCorrection = 1 - e
                                g *= gCorrection
                            if not self.ignoreClickNecessity:
                                b = self.cn[layout[rank]]
                                alphaFractions[intent][query][url][0] += a * (1 - b * g) / (1 - a * b * g) * p_I__C_G[
                                    intent]
                                self.getGamma(gammaFractions, rank, prevClick, layout, intent)[0] += \
                                    g / gCorrection * (1 - a * b) / (1 - a * b * g) * p_I__C_G[intent]
                                if self.explorationBias and explorationBiasPossible:
                                    eFractions[firstVerticalPos][0] += \
                                        (e if s.layout[k] else e / (1 - a * b * g)) * p_I__C_G[intent]
                                cnFractions[layout[rank]][0] += b * (1 - a * g) / (1 - a * b * g) * p_I__C_G[intent]
                            else:
                                alphaFractions[intent][query][url][0] += a * (1 - g) / (1 - a * g) * p_I__C_G[intent]
                                self.getGamma(gammaFractions, rank, prevClick, layout, intent)[0] += g / gCorrection * (
                                1 - a) / (1 - a * g) * p_I__C_G[intent]
                                if self.explorationBias and explorationBiasPossible:
                                    eFractions[firstVerticalPos][0] += (e if s.layout[k] else e / (1 - a * g)) * \
                                                                       p_I__C_G[intent]
                        else:
                            alphaFractions[intent][query][url][0] += 1 * p_I__C_G[intent]
                            self.getGamma(gammaFractions, rank, prevClick, layout, intent)[0] += 1 * p_I__C_G[intent]
                            if not self.ignoreClickNecessity:
                                cnFractions[layout[rank]][0] += 1 * p_I__C_G[intent]
                            if self.explorationBias and explorationBiasPossible:
                                eFractions[firstVerticalPos][0] += (e if s.layout[k] else 0) * p_I__C_G[intent]

                        alphaFractions[intent][query][url][1] += 1 * p_I__C_G[intent]
                        self.getGamma(gammaFractions, rank, prevClick, layout, intent)[1] += 1 * p_I__C_G[intent]
                        if not self.ignoreClickNecessity:
                            cnFractions[layout[rank]][1] += 1 * p_I__C_G[intent]
                        if self.explorationBias and explorationBiasPossible:
                            eFractions[firstVerticalPos][1] += 1 * p_I__C_G[intent]
                    if c != 0:
                        prevClick = rank
            if not self.config.get('PRETTY_LOG', PRETTY_LOG):
                sys.stderr.write('E')
            # M-step
            sum_square_displacement = 0.0
            for i in possibleIntents:
                for q in xrange(max_query_id):
                    for url, aF in alphaFractions[i][q].iteritems():
                        new_alpha = aF[0] / aF[1]
                        sum_square_displacement += (self.alpha[i][q][url] - new_alpha) ** 2
                        self.alpha[i][q][url] = new_alpha
            for g in xrange(self.gammaTypesNum):
                for r in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY)):
                    for d in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY)):
                        gF = gammaFractions[g][r][d]
                        new_gamma = gF[0] / gF[1]
                        sum_square_displacement += (self.gamma[g][r][d] - new_gamma) ** 2
                        self.gamma[g][r][d] = new_gamma
            if not self.ignoreClickNecessity:
                for vertical_id in xrange(self.config.get('MAX_VERTICAL_ID', MAX_VERTICAL_ID) + 1):
                    new_cn = cnFractions[vertical_id][0] / cnFractions[vertical_id][1]
                    sum_square_displacement += (self.cn[vertical_id] - new_cn) ** 2
                    self.cn[vertical_id] = new_cn
            if self.explorationBias:
                for p in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY)):
                    new_e = eFractions[p][0] / eFractions[p][1]
                    sum_square_displacement += (self.e[p] - new_e) ** 2
                    self.e[p] = new_e
            if not self.config.get('PRETTY_LOG', PRETTY_LOG):
                sys.stderr.write('M\n')
            rmsd = math.sqrt(sum_square_displacement)
            if self.config.get('PRETTY_LOG', PRETTY_LOG):
                sys.stderr.write('%d..' % (iteration_count + 1))
            else:
                print >> sys.stderr, 'Iteration: %d, ERROR: %f' % (iteration_count + 1, rmsd)
                if not self.ignoreClickNecessity:
                    print >> sys.stderr, 'Click Necessity', self.cn
        if self.config.get('PRETTY_LOG', PRETTY_LOG):
            sys.stderr.write('\n')
        for q, intentWeights in self.queryIntentsWeights.iteritems():
            self.queryIntentsWeights[q] = sum(intentWeights) / len(intentWeights)

    def _getSessionProb(self, s):
        clickProbs = self._get_click_probs(s, [False, True])
        N = len(s.clicks)
        return clickProbs[False][N - 1] / clickProbs[True][N - 1]

    @staticmethod
    def getGamma(gammas, k, prevClick, layout, intent):
        index = (2 if layout[k] else 0) + (1 if intent else 0)
        return gammas[index][k][k - prevClick - 1]

    @staticmethod
    def getGammaWithVerticalId(gammas, k, prevClick, layout, intent):
        index = 2 * int(layout[k]) + (1 if intent else 0)
        return gammas[index][k][k - prevClick - 1]

    def _get_click_probs(self, s, possibleIntents):
        """
            Returns clickProbs list
            clickProbs[i][k] = P(C_1, ..., C_k | I=i)
            """
        clickProbs = dict((i, []) for i in possibleIntents)
        firstVerticalPos = -1 if not any(s.layout[:-1]) else [k for (k, l) in enumerate(s.layout) if l][0]
        prevClick = -1
        layout = [0] * len(s.layout) if self.ignoreLayout else s.layout
        for rank, c in enumerate(s.clicks):
            url = s.results[rank]
            prob = {False: 0.0, True: 0.0}
            for i in possibleIntents:
                a = self.alpha[i][s.query][url]
                g = self.getGamma(self.gamma, rank, prevClick, layout, i)
                if self.explorationBias and any(s.layout[k] and s.clicks[k] for k in xrange(rank)) and not s.layout[
                    rank]:
                    g *= 1 - self.e[firstVerticalPos]
                prevProb = 1 if rank == 0 else clickProbs[i][-1]
                if not self.ignoreClickNecessity:
                    b = self.cn[layout[rank]]
                    if c == 0:
                        clickProbs[i].append(prevProb * (1 - a * b * g))
                    else:
                        clickProbs[i].append(prevProb * a * b * g)
                else:
                    if c == 0:
                        clickProbs[i].append(prevProb * (1 - a * g))
                    else:
                        clickProbs[i].append(prevProb * a * g)
            if c != 0:
                prevClick = rank
        return clickProbs

    def get_relevance_parameters(self, query, url, vertical_id):
        """return alpha, beta, s_c, s_e"""
        return self.alpha[False][query][url], 1.0, 0.0, 0.0


class EbUbmModel(UbmModel):
    def __init__(self, ignoreIntents=True, ignoreLayout=True, config=None):
        UbmModel.__init__(self, ignoreIntents, ignoreLayout, explorationBias=True,
                          config=config)


class DcmModel(ClickModel):
    gammaTypesNum = 4

    def train(self, sessions):
        max_query_id = self.config.get('MAX_QUERY_ID')
        if max_query_id is None:
            print >> sys.stderr, 'WARNING: no MAX_QUERY_ID specified for', self
            max_query_id = 100000
        possibleIntents = [False] if self.ignoreIntents else [False, True]
        urlRelFractions = dict(
            (i, [defaultdict(lambda: [1.0, 1.0]) for q in xrange(max_query_id)]) for i in possibleIntents)
        gammaFractions = [[[1.0, 1.0] for g in xrange(self.gammaTypesNum)] \
                          for r in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))]
        for s in sessions:
            query = s.query
            layout = [False] * len(s.layout) if self.ignoreLayout else s.layout
            lastClickedPos = self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY) - 1
            for k, c in enumerate(s.clicks):
                if c != 0:
                    lastClickedPos = k
            intentWeights = {False: 1.0} if self.ignoreIntents else {False: 1 - s.intentWeight, True: s.intentWeight}
            for k, (u, c) in enumerate(zip(s.results, s.clicks[:(lastClickedPos + 1)])):
                for i in possibleIntents:
                    if c != 0:
                        urlRelFractions[i][query][u][1] += intentWeights[i]
                        if k == lastClickedPos:
                            self.getGamma(gammaFractions[k], k, layout, i)[1] += intentWeights[i]
                        else:
                            self.getGamma(gammaFractions[k], k, layout, i)[0] += intentWeights[i]
                    else:
                        urlRelFractions[i][query][u][0] += intentWeights[i]
        self.urlRelevances = dict((i,
                                   [defaultdict(lambda: self.config.get('DEFAULT_REL', DEFAULT_REL)) \
                                    for q in xrange(max_query_id)]) for i in possibleIntents)
        self.gammas = [[0.5 for g in xrange(self.gammaTypesNum)] \
                       for r in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))]
        for i in possibleIntents:
            for query, d in enumerate(urlRelFractions[i]):
                if not d:
                    continue
                for url, relFractions in d.iteritems():
                    self.urlRelevances[i][query][url] = relFractions[1] / (relFractions[1] + relFractions[0])
        for k in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY)):
            for g in xrange(self.gammaTypesNum):
                self.gammas[k][g] = gammaFractions[k][g][0] / (gammaFractions[k][g][0] + gammaFractions[k][g][1])

    def _get_click_probs(self, s, possibleIntents):
        clickProbs = {False: [], True: []}  # P(C_1, ..., C_k)
        query = s.query
        layout = [False] * len(s.layout) if self.ignoreLayout else s.layout
        for i in possibleIntents:
            examinationProb = 1.0  # P(C_1, ..., C_{k - 1}, E_k = 1)
            for k, c in enumerate(s.clicks):
                r = self.urlRelevances[i][query][s.results[k]]
                prevProb = 1 if k == 0 else clickProbs[i][-1]
                if c == 0:
                    # P(C_1, ..., C_k = 0) = P(C_1, ..., C_{k-1}) - P(C_1, ..., C_k = 1)
                    clickProbs[i].append(prevProb - examinationProb * r)
                    # P(C_1, ..., C_k, E_{k+1} = 1) = P(E_{k+1} = 1 | C_k, E_k = 1) * P(C_k | E_k = 1) *  P(C_1, ..., C_{k - 1}, E_k = 1)
                    examinationProb *= 1 - r
                else:
                    clickProbs[i].append(examinationProb * r)
                    # P(C_1, ..., C_k, E_{k+1} = 1) = P(E_{k+1} = 1 | C_k, E_k = 1) * P(C_k | E_k = 1) *  P(C_1, ..., C_{k - 1}, E_k = 1)
                    examinationProb *= self.getGamma(self.gammas[k], k, layout, i) * r
        return clickProbs

    def get_relevance_parameters(self, query, url, vertical_id):
        """return alpha, beta, s_c, s_e"""
        return self.urlRelevances[False][query][url], 1.0, 0.0, 0.0

    @staticmethod
    def getGamma(gammas, k, layout, intent):
        return DbnModel.getGamma(gammas, k, layout, intent)


class McmModel(ClickModel):
    def __init__(self, ignoreIntents=True, ignoreLayout=False,
                 ignoreVerticalType=False, ignoreClickNecessity=False,
                 ignoreClickSatisfaction=False, ignoreExamSatisfaction=False,
                 useViewportTime=True,
                 config=None):

        self.ignoreVerticalType = ignoreVerticalType
        self.ignoreClickNecessity = ignoreClickNecessity
        self.ignoreClickSatisfaction = ignoreClickSatisfaction
        self.ignoreExamSatisfaction = ignoreExamSatisfaction
        self.useViewportTime = useViewportTime

        ClickModel.__init__(self, ignoreIntents, ignoreLayout, config)
        print >> sys.stderr, 'McmModel:' + \
                             ' ignoreLayout=' + str(self.ignoreLayout) + \
                             ' ignoreVerticalType=' + str(self.ignoreVerticalType) + \
                             ' ignoreClickNecessity=' + str(self.ignoreClickNecessity) + \
                             ' ignoreClickSatisfaction=' + str(self.ignoreClickSatisfaction) + \
                             ' ignoreExamSatisfaction=' + str(self.ignoreExamSatisfaction) + \
                             ' useViewportTime=' + str(self.useViewportTime)

        # does not support IA model
        assert ignoreIntents

        # default UBM gamma
        self.gammaTypesNum = 2
        # UBM-layout, separate gamma for each vertical type
        if self.ignoreClickNecessity and not self.ignoreVerticalType:
            self.gammaTypesNum = self.config.get('MAX_VERTICAL_ID', MAX_VERTICAL_ID)
            self.getGamma = self.getGammaWithVerticalId

    def train(self, sessions):
        # initialize alpha, beta, gamma, s_c, s_e #
        max_query_id = self.config.get('MAX_QUERY_ID')
        if max_query_id is None:
            print >> sys.stderr, 'WARNING: no MAX_QUERY_ID specified for', self
            max_query_id = 100000
        possibleIntents = [False] if self.ignoreIntents else [False, True]
        # alpha: intent -> query -> url -> "attractiveness probability"
        self.alpha = dict((i,
                           [defaultdict(lambda: self.config.get('DEFAULT_REL', DEFAULT_REL)) \
                            for q in xrange(max_query_id)]) for i in possibleIntents)
        # gamma: freshness of the current result: gammaType -> rank -> "distance from the last click" - 1 -> examination probability
        self.gamma = [[[0.5 \
                        for d in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))] \
                       for r in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))] \
                      for g in xrange(self.gammaTypesNum)]
        # beta: click necessity of the result, vertical_id -> probability that the result needs click
        if not self.ignoreClickNecessity:
            self.beta = [0.5 \
                         for vertical_id in xrange(self.config.get('MAX_VERTICAL_ID', MAX_VERTICAL_ID))]
        else:
            self.beta = [1.0 \
                         for vertical_id in xrange(self.config.get('MAX_VERTICAL_ID', MAX_VERTICAL_ID))]
        # s_c: prob. of satisfaction after click
        if not self.ignoreClickSatisfaction:
            self.s_c = dict((i,
                             [defaultdict(lambda: self.config.get('DEFAULT_SAT_CLICK', DEFAULT_SAT_CLICK)) \
                              for q in xrange(max_query_id)]) for i in possibleIntents)
        else:
            self.s_c = dict((i,
                             [defaultdict(lambda: 0.0) \
                              for q in xrange(max_query_id)]) for i in possibleIntents)

        # s_e: prob. of satisfaction after examining a result that does not need click
        if not self.ignoreExamSatisfaction:
            self.s_e = dict((i,
                             [defaultdict(lambda: self.config.get('DEFAULT_SAT_EXAM', DEFAULT_SAT_EXAM)) \
                              for q in xrange(max_query_id)]) for i in possibleIntents)
        else:
            self.s_e = dict((i,
                             [defaultdict(lambda: 0.0) \
                              for q in xrange(max_query_id)]) for i in possibleIntents)

        # start training #
        if not self.config.get('PRETTY_LOG', PRETTY_LOG):
            print >> sys.stderr, '-' * 80
            print >> sys.stderr, 'Start. Current time is', datetime.now()
        for iteration_count in xrange(self.config.get('MAX_ITERATIONS', MAX_ITERATIONS)):
            self.queryIntentsWeights = defaultdict(lambda: [])
            # not like in DBN! xxxFractions[0] is a numerator while xxxFraction[1] is a denominator
            alphaFractions = dict((i, [defaultdict(lambda: list(self.config.get('ALPHA_PRIOR', ALPHA_PRIOR)))
                                       for q in xrange(max_query_id)]) for i in possibleIntents)
            gammaFractions = [[[[1.0, 2.0]
                                for d in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))]
                               for r in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))]
                              for g in xrange(self.gammaTypesNum)]
            if not self.ignoreClickNecessity:
                betaFractions = [list(self.config.get('BETA_PRIOR', BETA_PRIOR))
                                 for vertical_id in xrange(self.config.get('MAX_VERTICAL_ID', MAX_VERTICAL_ID))]
            if not self.ignoreClickSatisfaction:
                s_cFractions = [defaultdict(lambda: list(self.config.get('S_C_PRIOR', S_C_PRIOR)))
                                for q in xrange(max_query_id)]
            if not self.ignoreExamSatisfaction:
                s_eFractions = [defaultdict(lambda: list(self.config.get('S_E_PRIOR', S_E_PRIOR)))
                                for q in xrange(max_query_id)]

            # E-step
            for s in sessions:
                query = s.query
                if self.useViewportTime:
                    vpts = s.extraclicks['w_vpt']
                else:
                    vpts = [0] * len(s.clicks)
                layout = [0] * len(s.layout) if self.ignoreLayout else s.layout
                alphaList, betaList, gammaList, s_cList, s_eList = self.get_session_parameters(s, layout)
                f, b, z = self.get_forward_backward(s.clicks, alphaList, betaList, gammaList, s_cList, s_eList)

                prevClick = -1
                for rank, (c, result) in enumerate(zip(s.clicks, s.results)):
                    # f_{i-1}
                    f0 = 1.0 if rank == 0 else f[rank - 1][0]
                    f1 = 0.0 if rank == 0 else f[rank - 1][1]
                    vpt = vpts[rank]
                    # update gamma
                    if c == 1:
                        # P(E_i=1, S_i-1=0|C_1..M) = 1
                        self.getGamma(gammaFractions, rank, prevClick, layout, False)[0] += 1.0
                        self.getGamma(gammaFractions, rank, prevClick, layout, False)[1] += 1.0
                    else:
                        # P(E_i=x, S_i-1=y|C_1..M) = pxy
                        p00 = f0 * b[rank][0] * (1.0 - gammaList[rank])
                        # P(E_i=x, S_e_i=y, S_i-1=0|C_1..M) = qxy
                        q10 = b[rank][0] * (1.0 - alphaList[rank] +
                                            alphaList[rank] * (1.0 - betaList[rank]) * (1.0 - s_eList[rank])) \
                                            * f0 * gammaList[rank]
                        q11 = b[rank][1] * alphaList[rank] * (1.0 - betaList[rank]) * s_eList[rank] \
                              * f0 * gammaList[rank]
                        p10 = q10 + q11
                        p01 = f1 * b[rank][1]

                        z_p = p00 + p10 + p01
                        self.getGamma(gammaFractions, rank, prevClick, layout, False)[0] += p10 / z_p
                        self.getGamma(gammaFractions, rank, prevClick, layout, False)[1] += 1.0 - p01 / z_p

                    # update alpha
                    if c == 1:
                        # P(R_i=1|C_1..M) = 1
                        alphaFractions[False][query][s.results[rank]][0] += 1.0
                    else:
                        # P(R_i=1|C_1..M) = p1
                        p1 = b[rank][0] * (1.0 - gammaList[rank] +
                                           gammaList[rank] * (1.0 - betaList[rank]) * (1.0 - s_eList[rank]))
                        p1 += b[rank][1] * gammaList[rank] * (1.0 - betaList[rank]) * s_eList[rank]
                        p1 *= f0
                        p1 += b[rank][1] * f1
                        p1 *= alphaList[rank]
                        p1 /= z[rank]
                        alphaFractions[False][query][s.results[rank]][0] += p1
                    alphaFractions[False][query][s.results[rank]][1] += 1.0

                    # update beta
                    if not self.ignoreClickNecessity:
                        if c == 1:
                            # P(N_i=1|C_1..M) = 1
                            betaFractions[s.layout[rank]][0] += 1.0
                        else:
                            # P(N_i=1|C_1..M) = p1
                            p1 = f0 * b[rank][0] * (1.0 - gammaList[rank] * alphaList[rank]) + f1 * b[rank][1]
                            p1 *= betaList[rank]
                            p1 /= z[rank]
                            betaFractions[s.layout[rank]][0] += p1
                        betaFractions[s.layout[rank]][1] += 1.0

                    # update s_c
                    if not self.ignoreClickSatisfaction:
                        if c == 1:
                            # P(E_i=1, R_i=1, N_i=1, S_i=1|C_1..M) = p1
                            # P(E_i=1, R_i=1, N_i=1, S_i=0|C_1..M) = p0
                            p1 = f0 * b[rank][1]
                            p1 *= gammaList[rank] * alphaList[rank] * betaList[rank] * s_cList[rank]
                            p0 = f0 * b[rank][0]
                            p0 *= gammaList[rank] * alphaList[rank] * betaList[rank] * (1.0 - s_cList[rank])
                            p1, p0 = p1 / z[rank], p0 / z[rank]
                            s_cFractions[query][s.results[rank]][0] += p1
                            s_cFractions[query][s.results[rank]][1] += p1 + p0

                    # update s_e
                    if not self.ignoreExamSatisfaction:
                        if c == 0:
                            # P(E_i=1, R_i=1, N_i=1, S_i=1|C_1..M) = p1
                            # P(E_i=1, R_i=1, N_i=1, S_i=0|C_1..M) = p0
                            p1 = f0 * b[rank][1]
                            p1 *= gammaList[rank] * alphaList[rank] * (1.0 - betaList[rank]) * s_eList[rank]
                            p0 = f0 * b[rank][0]
                            p0 *= gammaList[rank] * alphaList[rank] * (1.0 - betaList[rank]) * (1.0 - s_eList[rank])
                            p1, p0 = p1 / z[rank], p0 / z[rank]
                            s_eFractions[query][s.results[rank]][0] += p1
                            s_eFractions[query][s.results[rank]][1] += p1 + p0

                    # update prevClick
                    if c == 1:
                        prevClick = rank

            if not self.config.get('PRETTY_LOG', PRETTY_LOG):
                sys.stderr.write('E')

            # M-step
            sum_square_displacement = 0.0
            # gamma
            for g in xrange(self.gammaTypesNum):
                for r in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY)):
                    for d in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY)):
                        gF = gammaFractions[g][r][d]
                        new_gamma = gF[0] / gF[1]
                        sum_square_displacement += (self.gamma[g][r][d] - new_gamma) ** 2
                        self.gamma[g][r][d] = new_gamma
            # alpha
            for i in possibleIntents:
                for q in xrange(max_query_id):
                    for url, aF in alphaFractions[i][q].iteritems():
                        new_alpha = aF[0] / aF[1]
                        sum_square_displacement += (self.alpha[i][q][url] - new_alpha) ** 2
                        self.alpha[i][q][url] = new_alpha
            # beta
            if not self.ignoreClickNecessity:
                for vertical_id in xrange(self.config.get('MAX_VERTICAL_ID', MAX_VERTICAL_ID)):
                    new_cn = betaFractions[vertical_id][0] / betaFractions[vertical_id][1]
                    sum_square_displacement += (self.beta[vertical_id] - new_cn) ** 2
                    self.beta[vertical_id] = new_cn
            # s_c
            if not self.ignoreClickSatisfaction:
                for q in xrange(max_query_id):
                    for url, s_cF in s_cFractions[q].iteritems():
                        new_s_c = s_cF[0] / s_cF[1]
                        sum_square_displacement += (self.s_c[False][q][url] - new_s_c) ** 2
                        self.s_c[False][q][url] = new_s_c
            # s_e
            if not self.ignoreExamSatisfaction:
                for q in xrange(max_query_id):
                    for url, s_eF in s_eFractions[q].iteritems():
                        new_s_e = s_eF[0] / s_eF[1]
                        sum_square_displacement += (self.s_e[False][q][url] - new_s_e) ** 2
                        self.s_e[False][q][url] = new_s_e

            if not self.config.get('PRETTY_LOG', PRETTY_LOG):
                sys.stderr.write('M\n')
            rmsd = math.sqrt(sum_square_displacement)
            if self.config.get('PRETTY_LOG', PRETTY_LOG):
                sys.stderr.write('%d..' % (iteration_count + 1))
            else:
                print >> sys.stderr, 'Iteration: %d, ERROR: %f' % (iteration_count + 1, rmsd)
        if self.config.get('PRETTY_LOG', PRETTY_LOG):
            sys.stderr.write('\n')

    def get_session_parameters(self, session, layout):
        """return alphaList, betaList, gammaList, s_cList, s_eList"""
        alphaList = []
        betaList = []
        gammaList = []
        s_cList = []
        s_eList = []

        query = session.query
        prevClick = -1
        lastClick = max([-1] + [rank for rank, c in enumerate(session.clicks) if c])
        for rank, url in enumerate(session.results):
            alphaList.append(self.alpha[False][query][url])

            gammaList.append(self.getGamma(self.gamma, rank, prevClick, layout, False))
            if session.clicks[rank]:
                prevClick = rank

            betaList.append(self.beta[layout[rank]])

            s_cList.append(self.s_c[False][query][url])
            s_eList.append(self.s_e[False][query][url])

        return alphaList, betaList, gammaList, s_cList, s_eList

    def get_relevance_parameters(self, query, url, vertical_id):
        """return alpha, beta, s_c, s_e"""
        return self.alpha[False][query][url], \
               self.beta[vertical_id], \
               self.s_c[False][query][url], \
               self.s_e[False][query][url]

    @staticmethod
    def get_forward_backward(clicks, alphaList, betaList, gammaList, s_cList, s_eList):
        """return f, b, z"""
        M = len(clicks)
        f = [[1.0, 0]]
        b = [[1.0, 1.0] for i in xrange(M)]
        z = [1.0]

        # P(S_i=t, C_i|S_i-1=0) = p[i][t]
        p = []

        # forward
        for i, c in enumerate(clicks):
            if c == 0:
                t0 = 1.0 - gammaList[i]
                t0 += gammaList[i] * \
                      (1.0 - alphaList[i] + alphaList[i] * (1.0 - betaList[i]) * (1.0 - s_eList[i]))
                t1 = gammaList[i] * alphaList[i] * (1.0 - betaList[i]) * s_eList[i]
                p.append([t0, t1])

                f.append([
                    f[i][0] * p[i][0],
                    f[i][0] * p[i][1] + f[i][1]
                ])
            else:
                p.append([
                    gammaList[i] * alphaList[i] * betaList[i] * (1.0 - s_cList[i]),
                    gammaList[i] * alphaList[i] * betaList[i] * s_cList[i]
                ])

                f.append([
                    f[i][0] * p[i][0],
                    f[i][0] * p[i][1]
                ])

            z.append(sum(f[i + 1]))

            f[i + 1][0] /= z[i + 1]
            f[i + 1][1] /= z[i + 1]

        f = f[1:]
        z = z[1:]

        # backward
        # re-used p[i][t] computed in the forward pass
        for i in range(M - 2, -1, -1):
            if clicks[i + 1] == 0:
                b[i][0] = b[i + 1][0] * p[i + 1][0] + b[i + 1][1] * p[i + 1][1]
                b[i][1] = b[i + 1][1]
            else:
                b[i][0] = b[i + 1][0] * p[i + 1][0] + b[i + 1][1] * p[i + 1][1]
                b[i][1] = 0.0
            b[i] = [b[i][0] / z[i + 1], b[i][1] / z[i + 1]]

        return f, b, z

    @staticmethod
    def test_forward_backward(clicks=[1, 0, 0, 1, 0]):
        M = len(clicks)
        alphaList = [0.5] * M
        print 'alpha: ' + ' '.join(["%.3f" % x for x in alphaList])
        betaList = [0.5] * M
        print 'beta: ' + ' '.join(["%.3f" % x for x in betaList])
        gammaList = list(reversed([0.1 * i for i in range(1, M + 1)]))
        print 'gamma: ' + ' '.join(["%.3f" % x for x in gammaList])
        s_cList = [0.8] * M
        print 's_cList: ' + ' '.join(["%.3f" % x for x in s_cList])
        s_eList = [0.5] * M
        print 's_eList: ' + ' '.join(["%.3f" % x for x in s_eList])
        f, b, z = McmModel.get_forward_backward(clicks, alphaList, betaList, gammaList, s_cList, s_eList)
        print 'results:'
        print 'f0:\t' + ' '.join(["%.3f" % x[0] for x in f])
        print 'f1:\t' + ' '.join(["%.3f" % x[1] for x in f])
        print 'z:\t' + ' '.join(["%.3f" % x for x in z])

        print 'b0:\t' + ' '.join(["%.3f" % x[0] for x in b])
        print 'b1:\t' + ' '.join(["%.3f" % x[1] for x in b])

    def _getSessionProb(self, s):
        clickProbs = self._get_click_probs(s, [False, True])
        N = len(s.clicks)
        return clickProbs[False][N - 1] / clickProbs[True][N - 1]

    @staticmethod
    def getGamma(gammas, k, prevClick, layout, intent):
        index = 1 if layout[k] else 0
        return gammas[index][k][k - prevClick - 1]

    @staticmethod
    def getGammaWithVerticalId(gammas, k, prevClick, layout, intent):
        index = int(layout[k])
        return gammas[index][k][k - prevClick - 1]

    def _get_click_probs(self, s, possibleIntents):
        """
            Returns clickProbs list
            clickProbs[i][k] = P(C_1, ..., C_k | I=i)
            """
        alpha, beta, gamma, s_c, s_e = self.get_session_parameters(s, s.layout)
        f, b, z = self.get_forward_backward(s.clicks, alpha, beta, gamma, s_c, s_e)
        clickProbs = []
        for i in xrange(len(z)):
            clickProbs.append(1.0 * z[i] if i == 0 else clickProbs[i - 1] * z[i])
        return dict((i, clickProbs) for i in possibleIntents)


class McmVptModel(ClickModel):
    def __init__(self, ignoreIntents=True, ignoreLayout=False,
                 ignoreVerticalType=False, ignoreClickNecessity=False,
                 ignoreClickSatisfaction=False, ignoreExamSatisfaction=False,
                 useViewportTime=True, viewport_time_model=0,
                 config=None):

        self.ignoreVerticalType = ignoreVerticalType
        self.ignoreClickNecessity = ignoreClickNecessity
        self.ignoreClickSatisfaction = ignoreClickSatisfaction
        self.ignoreExamSatisfaction = ignoreExamSatisfaction
        self.useViewportTime = useViewportTime
        self.viewport_time_model = viewport_time_model

        ClickModel.__init__(self, ignoreIntents, ignoreLayout, config)
        print >> sys.stderr, 'McmVptModel:' + \
                             ' ignoreLayout=' + str(self.ignoreLayout) + \
                             ' ignoreVerticalType=' + str(self.ignoreVerticalType) + \
                             ' ignoreClickNecessity=' + str(self.ignoreClickNecessity) + \
                             ' ignoreClickSatisfaction=' + str(self.ignoreClickSatisfaction) + \
                             ' ignoreExamSatisfaction=' + str(self.ignoreExamSatisfaction) + \
                             ' useViewportTime=' + str(self.useViewportTime) + \
                             ' viewportTimeModel=' + str(self.viewport_time_model)

        # does not support IA model
        assert ignoreIntents
        assert useViewportTime
        if useViewportTime:
            if self.viewport_time_model == 0:
                self.viewport_time_model = NormalViewportTimeModel(
                    self.config.get('MAX_VERTICAL_ID', MAX_VERTICAL_ID))
                print >> sys.stderr, 'VIEWPORT_TIME_MODEL:', 'Log-normal'
            elif self.viewport_time_model == 1:
                self.viewport_time_model = GammaViewportTimeModel(
                    self.config.get('MAX_VERTICAL_ID', MAX_VERTICAL_ID))
                print >> sys.stderr, 'VIEWPORT_TIME_MODEL:', 'Gamma'
            elif self.viewport_time_model == 2:
                self.viewport_time_model = WeibullViewportTimeModel(
                    self.config.get('MAX_VERTICAL_ID', MAX_VERTICAL_ID))
                print >> sys.stderr, 'VIEWPORT_TIME_MODEL:', 'Weibull'
            elif self.viewport_time_model == 3:
                self.viewport_time_model = ComplexNormalViewportTimeModel(
                    self.config.get('MAX_VERTICAL_ID', MAX_VERTICAL_ID))
                print >> sys.stderr, 'VIEWPORT_TIME_MODEL:', 'Complex Log-normal'
            elif self.viewport_time_model == 4:
                self.viewport_time_model = ComplexGammaViewportTimeModel(
                    self.config.get('MAX_VERTICAL_ID', MAX_VERTICAL_ID))
                print >> sys.stderr, 'VIEWPORT_TIME_MODEL:', 'Complex Gamma'
            elif self.viewport_time_model == 5:
                self.viewport_time_model = ComplexWeibullViewportTimeModel(
                    self.config.get('MAX_VERTICAL_ID', MAX_VERTICAL_ID))
                print >> sys.stderr, 'VIEWPORT_TIME_MODEL:', 'Complex Weibull'
            elif self.viewport_time_model == 6:
                self.viewport_time_model = SimpleViewportTimeModel()
                print >> sys.stderr, 'VIEWPORT_TIME_MODEL:', 'Simple'
            elif self.viewport_time_model == 7:
                self.viewport_time_model = DefaultViewportTimeModel()
                print >> sys.stderr, 'VIEWPORT_TIME_MODEL:', 'Default'


        # default UBM gamma
        self.gammaTypesNum = 2
        # UBM-layout, separate gamma for each vertical type
        if self.ignoreClickNecessity and not self.ignoreVerticalType:
            self.gammaTypesNum = self.config.get('MAX_VERTICAL_ID', MAX_VERTICAL_ID)
            self.getGamma = self.getGammaWithVerticalId

    def train(self, sessions):
        # initialize alpha, beta, gamma, s_c, s_e #
        max_query_id = self.config.get('MAX_QUERY_ID')
        if max_query_id is None:
            print >> sys.stderr, 'WARNING: no MAX_QUERY_ID specified for', self
            max_query_id = 100000
        possibleIntents = [False] if self.ignoreIntents else [False, True]
        # alpha: intent -> query -> url -> "attractiveness probability"
        self.alpha = dict((i,
                           [defaultdict(lambda: self.config.get('DEFAULT_REL', DEFAULT_REL)) \
                            for q in xrange(max_query_id)]) for i in possibleIntents)
        # gamma: freshness of the current result: gammaType -> rank -> "distance from the last click" - 1 -> examination probability
        self.gamma = [[[0.5 \
                        for d in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))] \
                       for r in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))] \
                      for g in xrange(self.gammaTypesNum)]
        # beta: click necessity of the result, vertical_id -> probability that the result needs click
        if not self.ignoreClickNecessity:
            self.beta = [0.5 \
                         for vertical_id in xrange(self.config.get('MAX_VERTICAL_ID', MAX_VERTICAL_ID))]
        else:
            self.beta = [1.0 \
                         for vertical_id in xrange(self.config.get('MAX_VERTICAL_ID', MAX_VERTICAL_ID))]
        # s_c: prob. of satisfaction after click
        if not self.ignoreClickSatisfaction:
            self.s_c = dict((i,
                             [defaultdict(lambda: self.config.get('DEFAULT_SAT_CLICK', DEFAULT_SAT_CLICK)) \
                              for q in xrange(max_query_id)]) for i in possibleIntents)
        else:
            self.s_c = dict((i,
                             [defaultdict(lambda: 0.0) \
                              for q in xrange(max_query_id)]) for i in possibleIntents)

        # s_e: prob. of satisfaction after examining a result that does not need click
        if not self.ignoreExamSatisfaction:
            self.s_e = dict((i,
                             [defaultdict(lambda: self.config.get('DEFAULT_SAT_EXAM', DEFAULT_SAT_EXAM)) \
                              for q in xrange(max_query_id)]) for i in possibleIntents)
        else:
            self.s_e = dict((i,
                             [defaultdict(lambda: 0.0) \
                              for q in xrange(max_query_id)]) for i in possibleIntents)

        # start training #
        if not self.config.get('PRETTY_LOG', PRETTY_LOG):
            print >> sys.stderr, '-' * 80
            print >> sys.stderr, 'Start. Current time is', datetime.now()
        for iteration_count in xrange(self.config.get('MAX_ITERATIONS', MAX_ITERATIONS)):
            self.queryIntentsWeights = defaultdict(lambda: [])
            # not like in DBN! xxxFractions[0] is a numerator while xxxFraction[1] is a denominator
            alphaFractions = dict((i, [defaultdict(lambda: list(self.config.get('ALPHA_PRIOR', ALPHA_PRIOR)))
                                       for q in xrange(max_query_id)]) for i in possibleIntents)
            gammaFractions = [[[[1.0, 2.0]
                                for d in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))]
                               for r in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY))]
                              for g in xrange(self.gammaTypesNum)]
            if not self.ignoreClickNecessity:
                betaFractions = [list(self.config.get('BETA_PRIOR', BETA_PRIOR))
                                 for vertical_id in xrange(self.config.get('MAX_VERTICAL_ID', MAX_VERTICAL_ID))]
            if not self.ignoreClickSatisfaction:
                s_cFractions = [defaultdict(lambda: list(self.config.get('S_C_PRIOR', S_C_PRIOR)))
                                for q in xrange(max_query_id)]
            if not self.ignoreExamSatisfaction:
                s_eFractions = [defaultdict(lambda: list(self.config.get('S_E_PRIOR', S_E_PRIOR)))
                                for q in xrange(max_query_id)]
            self.viewport_time_model.update_init()

            # E-step
            for s in sessions:
                query = s.query
                if self.useViewportTime:
                    vpts = s.extraclicks['w_vpt']
                else:
                    vpts = [0] * len(s.clicks)
                layout = [0] * len(s.layout) if self.ignoreLayout else s.layout
                alphaList, betaList, gammaList, s_cList, s_eList = self.get_session_parameters(s, layout)
                f, b, z = self.get_forward_backward(s.clicks, alphaList, betaList, gammaList, s_cList, s_eList,
                                                    viewport_time=vpts, layout=layout)

                prevClick = -1
                for rank, (c, result) in enumerate(zip(s.clicks, s.results)):
                    vpt = vpts[rank]
                    # f_{i-1}
                    f0 = 1.0 if rank == 0 else f[rank - 1][0]
                    f1 = 0.0 if rank == 0 else f[rank - 1][1]

                    # update gamma
                    if c == 1:
                        # P(E_i=1, S_i-1=0|C_1..M) = 1
                        self.getGamma(gammaFractions, rank, prevClick, layout, False)[0] += 1.0
                        self.getGamma(gammaFractions, rank, prevClick, layout, False)[1] += 1.0
                        self.viewport_time_model.update(1.0, vpt, E=1, C=1, layout=layout[rank])
                        # self.viewport_time_model.update(0.0, vpt, E=0, C=1, layout=layout[rank])
                    else:
                        # P(E_i=x, S_i-1=y|C_1..M) = pxy
                        p00 = f0 * b[rank][0] * (1.0 - gammaList[rank]) \
                              * self.viewport_time_model.P(vpt, E=0, C=0, layout=layout[rank])
                        # P(E_i=x, S_e_i=y, S_i-1=0|C_1..M) = qxy
                        q10 = b[rank][0] * (1.0 - alphaList[rank] +
                                            alphaList[rank] * (1.0 - betaList[rank]) * (1.0 - s_eList[rank])) \
                                         * self.viewport_time_model.P(vpt, E=1, S_e=0, layout=layout[rank]) \
                                         * f0 * gammaList[rank]
                        q11 = b[rank][1] * alphaList[rank] * (1.0 - betaList[rank]) * s_eList[rank] \
                                         * self.viewport_time_model.P(vpt, E=1, S_e=1, layout=layout[rank]) \
                                         * f0 * gammaList[rank]
                        p10 = q10 + q11
                        p01 = f1 * b[rank][1] * self.viewport_time_model.P(vpt, E=0, C=0, layout=layout[rank])

                        z_p = p00 + p10 + p01
                        p1 = p10 / z_p
                        p0 = p00 / z_p
                        self.getGamma(gammaFractions, rank, prevClick, layout, False)[0] += p1
                        self.getGamma(gammaFractions, rank, prevClick, layout, False)[1] += p0 + p1

                        q10 /= z_p
                        q11 /= z_p

                        p00_nov = f0 * b[rank][0] * (1.0 - gammaList[rank])
                        q10_nov = b[rank][0] * (1.0 - alphaList[rank] +
                                            alphaList[rank] * (1.0 - betaList[rank]) * (1.0 - s_eList[rank])) * f0 * gammaList[rank]
                        q11_nov = b[rank][1] * alphaList[rank] * (1.0 - betaList[rank]) * s_eList[rank] * f0 * gammaList[rank]
                        p01_nov = f1 * b[rank][1]
                        z_p_nov = p00_nov + q10_nov + q11_nov + p01_nov

                        if 'complex' in self.viewport_time_model.get_method_name().lower():
                            self.viewport_time_model.update(q10, vpt, E=1, S_e=0, layout=layout[rank])
                            self.viewport_time_model.update(q11, vpt, E=1, S_e=1, layout=layout[rank])
                        else:
                            self.viewport_time_model.update(p1, vpt, E=1, layout=layout[rank])
                        self.viewport_time_model.update(1.0 - p1, vpt, E=0, C=0, layout=layout[rank])

                    # update alpha
                    if c == 1:
                        # P(R_i=1|C_1..M) = 1
                        alphaFractions[False][query][s.results[rank]][0] += 1.0
                    else:
                        # P(R_i=1|C_1..M) = p1
                        # S_i = 0
                        p1 = (1.0 - gammaList[rank]) * self.viewport_time_model.P(vpt, E=0, C=0, layout=layout[rank])
                        p1 += gammaList[rank] * self.viewport_time_model.P(vpt, E=1, S_e=0, layout=layout[rank]) * \
                              (1.0 - betaList[rank]) * (1.0 - s_eList[rank])
                        p1 *= b[rank][0]
                        # S_i = 1
                        p1 += b[rank][1] * gammaList[rank] * self.viewport_time_model.P(vpt, E=1, S_e=1, layout=layout[rank]) * \
                              (1.0 - betaList[rank]) * s_eList[rank]
                        # S_i-1 = 0
                        p1 *= f0
                        # S_i-1 = 1, S_i = 1
                        p1 += b[rank][1] * f1 * self.viewport_time_model.P(vpt, E=0, C=0, layout=layout[rank])
                        # R_i = 1
                        p1 *= alphaList[rank]
                        p1 /= z[rank]
                        alphaFractions[False][query][s.results[rank]][0] += p1
                    alphaFractions[False][query][s.results[rank]][1] += 1.0

                    # update beta
                    if not self.ignoreClickNecessity:
                        if c == 1:
                            # P(N_i=1|C_1..M) = 1
                            betaFractions[s.layout[rank]][0] += 1.0
                        else:
                            # P(N_i=1|C_1..M) = p1
                            p1 = f0 * b[rank][0]
                            p1 *= (1.0 - gammaList[rank]) * self.viewport_time_model.P(vpt, E=0, C=0, layout=layout[rank]) + \
                                  gammaList[rank] * self.viewport_time_model.P(vpt, E=1, S_e=0, layout=layout[rank]) * (1.0 - alphaList[rank])
                            p1 += f1 * b[rank][1] * self.viewport_time_model.P(vpt, E=0, C=0, layout=layout[rank])
                            p1 *= betaList[rank]
                            p1 /= z[rank]
                            betaFractions[s.layout[rank]][0] += p1
                        betaFractions[s.layout[rank]][1] += 1.0

                    # update s_c
                    if not self.ignoreClickSatisfaction:
                        if c == 1:
                            # P(E_i=1, R_i=1, N_i=1, S_i=1|C_1..M) = p1
                            # P(E_i=1, R_i=1, N_i=1, S_i=0|C_1..M) = p0
                            p1 = f0 * b[rank][1] * alphaList[rank] * betaList[rank] * s_cList[rank]
                            p1 *= gammaList[rank] * self.viewport_time_model.P(vpt, E=1, C=1, layout=layout[rank])
                            p0 = f0 * b[rank][0] * alphaList[rank] * betaList[rank] * (1.0 - s_cList[rank])
                            p0 *= gammaList[rank] * self.viewport_time_model.P(vpt, E=1, C=1, layout=layout[rank])
                            p1, p0 = p1 / z[rank], p0 / z[rank]
                            s_cFractions[query][s.results[rank]][0] += p1
                            s_cFractions[query][s.results[rank]][1] += p1 + p0

                    # update s_e
                    if not self.ignoreExamSatisfaction:
                        if c == 0:
                            # P(E_i=1, R_i=1, N_i=1, S_i=1|C_1..M) = p1
                            # P(E_i=1, R_i=1, N_i=1, S_i=0|C_1..M) = p0
                            p1 = f0 * b[rank][1] * alphaList[rank] * (1.0 - betaList[rank]) * s_eList[rank]
                            p1 *= gammaList[rank] * self.viewport_time_model.P(vpt, E=1, C=0, S_e=1, layout=layout[rank])
                            p0 = f0 * b[rank][0] * alphaList[rank] * (1.0 - betaList[rank]) * (1.0 - s_eList[rank])
                            p0 *= gammaList[rank] * self.viewport_time_model.P(vpt, E=1, C=0, S_e=0, layout=layout[rank])
                            p1, p0 = p1 / z[rank], p0 / z[rank]
                            s_eFractions[query][s.results[rank]][0] += p1
                            s_eFractions[query][s.results[rank]][1] += p1 + p0
                            # self.viewport_time_model.update(p1, vpt, S_e=1)
                            # self.viewport_time_model.update(1.0 - p1, vpt, S_e=0)

                    # update prevClick
                    if c == 1:
                        prevClick = rank

            if not self.config.get('PRETTY_LOG', PRETTY_LOG):
                sys.stderr.write('E')

            # M-step
            sum_square_displacement = 0.0
            # gamma
            for g in xrange(self.gammaTypesNum):
                for r in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY)):
                    for d in xrange(self.config.get('MAX_DOCS_PER_QUERY', MAX_DOCS_PER_QUERY)):
                        gF = gammaFractions[g][r][d]
                        new_gamma = gF[0] / gF[1]
                        sum_square_displacement += (self.gamma[g][r][d] - new_gamma) ** 2
                        self.gamma[g][r][d] = new_gamma
            # alpha
            for i in possibleIntents:
                for q in xrange(max_query_id):
                    for url, aF in alphaFractions[i][q].iteritems():
                        new_alpha = aF[0] / aF[1]
                        sum_square_displacement += (self.alpha[i][q][url] - new_alpha) ** 2
                        self.alpha[i][q][url] = new_alpha
            # beta
            if not self.ignoreClickNecessity:
                for vertical_id in xrange(self.config.get('MAX_VERTICAL_ID', MAX_VERTICAL_ID)):
                    new_cn = betaFractions[vertical_id][0] / betaFractions[vertical_id][1]
                    sum_square_displacement += (self.beta[vertical_id] - new_cn) ** 2
                    self.beta[vertical_id] = new_cn
            # s_c
            if not self.ignoreClickSatisfaction:
                for q in xrange(max_query_id):
                    for url, s_cF in s_cFractions[q].iteritems():
                        new_s_c = s_cF[0] / s_cF[1]
                        sum_square_displacement += (self.s_c[False][q][url] - new_s_c) ** 2
                        self.s_c[False][q][url] = new_s_c
            # s_e
            if not self.ignoreExamSatisfaction:
                for q in xrange(max_query_id):
                    for url, s_eF in s_eFractions[q].iteritems():
                        new_s_e = s_eF[0] / s_eF[1]
                        sum_square_displacement += (self.s_e[False][q][url] - new_s_e) ** 2
                        self.s_e[False][q][url] = new_s_e
            # update viewport model
            self.viewport_time_model.update_finalize()

            if not self.config.get('PRETTY_LOG', PRETTY_LOG):
                sys.stderr.write('M\n')
            rmsd = math.sqrt(sum_square_displacement)
            if self.config.get('PRETTY_LOG', PRETTY_LOG):
                sys.stderr.write('%d..' % (iteration_count + 1))
            else:
                print >> sys.stderr, 'Iteration: %d, ERROR: %f' % (iteration_count + 1, rmsd)
        if self.config.get('PRETTY_LOG', PRETTY_LOG):
            sys.stderr.write('\n')

    def get_session_parameters(self, session, layout):
        """return alphaList, betaList, gammaList, s_cList, s_eList"""
        alphaList = []
        betaList = []
        gammaList = []
        s_cList = []
        s_eList = []

        query = session.query
        prevClick = -1
        lastClick = max([-1] + [rank for rank, c in enumerate(session.clicks) if c])
        for rank, url in enumerate(session.results):
            alphaList.append(self.alpha[False][query][url])

            gammaList.append(self.getGamma(self.gamma, rank, prevClick, layout, False))
            if session.clicks[rank]:
                prevClick = rank

            betaList.append(self.beta[layout[rank]])

            s_cList.append(self.s_c[False][query][url])
            s_eList.append(self.s_e[False][query][url])

        return alphaList, betaList, gammaList, s_cList, s_eList

    def get_relevance_parameters(self, query, url, vertical_id):
        """return alpha, beta, s_c, s_e"""
        return self.alpha[False][query][url], \
               self.beta[vertical_id], \
               self.s_c[False][query][url], \
               self.s_e[False][query][url]

    def get_forward_backward(self, clicks, alphaList, betaList, gammaList, s_cList, s_eList, **kwargs):
        """return f, b, z"""
        M = len(clicks)
        vpts = kwargs['viewport_time']
        layouts = kwargs['layout']

        f = [[1.0, 0.]]
        b = [[1.0, 1.0] for i in xrange(M)]
        z = [1.0]

        # P(S_i=t, C_i|S_i-1=0) = p[i][t]
        p = []

        # forward
        for i, c in enumerate(clicks):
            vpt = vpts[i]
            layout = layouts[i]
            if c == 0:
                t0 = (1.0 - gammaList[i]) * self.viewport_time_model.P(vpt, E=0, C=0, layout=layout)
                t0 += gammaList[i] * self.viewport_time_model.P(vpt, E=1, S_e=0, layout=layout) * \
                      (1.0 - alphaList[i] + alphaList[i] * (1.0 - betaList[i]) * (1.0 - s_eList[i]))
                t1 = gammaList[i] * alphaList[i] * (1.0 - betaList[i]) * s_eList[i] * \
                     self.viewport_time_model.P(vpt, E=1, S_e=1, layout=layout)
                p.append([t0, t1])

                f.append([
                    f[i][0] * p[i][0],
                    f[i][0] * p[i][1] + f[i][1] * self.viewport_time_model.P(vpt, E=0, C=0, layout=layout)
                ])
            else:
                p.append([
                    gammaList[i] * self.viewport_time_model.P(vpt, E=1, C=1, layout=layout) * \
                    alphaList[i] * betaList[i] * (1.0 - s_cList[i]),
                    gammaList[i] * self.viewport_time_model.P(vpt, E=1, C=1, layout=layout) * \
                    alphaList[i] * betaList[i] * s_cList[i]
                ])

                f.append([
                    f[i][0] * p[i][0],
                    f[i][0] * p[i][1]
                ])

            z.append(sum(f[i + 1]))

            f[i + 1][0] /= z[i + 1]
            f[i + 1][1] /= z[i + 1]

        f = f[1:]
        z = z[1:]

        # backward
        # re-used p[i][t] computed in the forward pass
        for i in range(M - 2, -1, -1):

            if clicks[i + 1] == 0:
                b[i][0] = b[i + 1][0] * p[i + 1][0] + b[i + 1][1] * p[i + 1][1]
                b[i][1] = b[i + 1][1] * self.viewport_time_model.P(vpts[i + 1], E=0, C=0, layout=layouts[i+1])
            else:
                b[i][0] = b[i + 1][0] * p[i + 1][0] + b[i + 1][1] * p[i + 1][1]
                b[i][1] = 0.0
            b[i] = [b[i][0] / z[i + 1], b[i][1] / z[i + 1]]

        return f, b, z

    def test_forward_backward(self, clicks=[1, 0, 0, 1, 0], viewport_time=[1000, 1000, 1000, 1000, 0]):
        M = len(clicks)
        alphaList = [0.5] * M
        print 'alpha: ' + ' '.join(["%.3f" % x for x in alphaList])
        betaList = [0.5] * M
        print 'beta: ' + ' '.join(["%.3f" % x for x in betaList])
        gammaList = list(reversed([0.1 * i for i in range(1, M + 1)]))
        print 'gamma: ' + ' '.join(["%.3f" % x for x in gammaList])
        s_cList = [0.8] * M
        print 's_cList: ' + ' '.join(["%.3f" % x for x in s_cList])
        s_eList = [0.5] * M
        print 's_eList: ' + ' '.join(["%.3f" % x for x in s_eList])
        f, b, z = self.get_forward_backward(clicks, alphaList, betaList, gammaList, s_cList, s_eList,
                                            viewport_time=viewport_time)
        print 'results:'
        print 'f0:\t' + ' '.join(["%.3f" % x[0] for x in f])
        print 'f1:\t' + ' '.join(["%.3f" % x[1] for x in f])
        print 'z:\t' + ' '.join(["%.3f" % x for x in z])

        print 'b0:\t' + ' '.join(["%.3f" % x[0] for x in b])
        print 'b1:\t' + ' '.join(["%.3f" % x[1] for x in b])

    def _getSessionProb(self, s):
        clickProbs = self._get_click_probs(s, [False, True])
        N = len(s.clicks)
        return clickProbs[False][N - 1] / clickProbs[True][N - 1]

    @staticmethod
    def getGamma(gammas, k, prevClick, layout, intent):
        index = 1 if layout[k] else 0
        return gammas[index][k][k - prevClick - 1]

    @staticmethod
    def getGammaWithVerticalId(gammas, k, prevClick, layout, intent):
        index = int(layout[k])
        return gammas[index][k][k - prevClick - 1]

    def _get_click_probs(self, s, possibleIntents):
        """
            Returns clickProbs list
            clickProbs[i][k] = P(C_1, ..., C_k | I=i)
        """

        vpts = [0] * len(s.clicks)

        if self.useViewportTime:
            vpts = s.extraclicks['w_vpt']

        layout = [0] * len(s.layout) if self.ignoreLayout else s.layout
        alpha, beta, gamma, s_c, s_e = self.get_session_parameters(s, layout)
        f, b, z = self.get_forward_backward(s.clicks, alpha, beta, gamma, s_c, s_e, viewport_time=vpts, layout=layout)

        clickProbs = [1.0]
        for i, c in enumerate(s.clicks):
            f0, f1 = (f[i - 1][0], f[i - 1][1]) if i >= 1 else (1.0, 0.0)
            # P(E_i=1, C_i=1, V=vpt|S_i-1=0)
            p_e1_c1 = gamma[i] * self.viewport_time_model.P(vpts[i], E=1, C=1, layout=layout[i]) * alpha[i] * beta[i]
            # P(E_i=1, C_i=0, V=vpt|S_i-1=0)
            p_e1_c0 = gamma[i] * self.viewport_time_model.P(vpts[i], E=1, S_e=0, layout=layout[i]) * \
                    (1 - alpha[i] + alpha[i] * (1 - beta[i]) * (1 - s_e[i]))
            p_e1_c0 += gamma[i] * self.viewport_time_model.P(vpts[i], E=1, S_e=1, layout=layout[i]) * \
                     alpha[i] * (1 - beta[i]) * s_e[i]

            # P(E_i=1, V=vpt|C_1..i-1, V_1...i-1)
            p_e1 = f0 * (p_e1_c0 + p_e1_c1)
            # P(E_i=0, V=vpt|C_1..i-1, V_1...i-1)
            p_e0 = (f1 + f0 * (1 - gamma[i])) * self.viewport_time_model.P(vpts[i], E=0, C=0, layout=layout[i])
            # P(E_i=1|C_1..i-1, V_1...i)
            p_e = p_e1 / (p_e1 + p_e0)

            if c == 1:
                p_c = p_e * alpha[i] * beta[i]
            else:
                p_c = 1.0 - p_e * alpha[i] * beta[i]
            clickProbs.append(clickProbs[-1] * p_c)
        return dict((i, clickProbs[1:]) for i in possibleIntents)
