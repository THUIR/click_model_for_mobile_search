import numpy as np
import math

from .config_sample import *

class DefaultViewportTimeModel(object):
    def __init__(self):
        pass

    def P(self, v, **kwargs):
        """
        compute P(v|conditions)
        Args:
            v: viewport time
            **kwargs: conditions such as {'E': 1, 'C': 0}, which means

        Returns: P(v|conditions)

        """
        return 1.0

    def update_init(self):
        """
        initialize the training parameters before M-step
        Returns: None

        """
        pass

    def update(self, posterior, v, **kwargs):
        """
        update the training parameters when there is an item in the Q-function of the form
        P(conditions|C_1..M, V_1..M) * log P(V_i|conditions)
        Args:
            posterior:  posterior prob P(conditions|C_1..M, V_1..M) estimated in E-step
            v: viewport time
            **kwargs: conditions su

        Returns: None

        """
        pass

    def update_finalize(self):
        """
        update the model parameters using the training parameters
        Returns: None

        """
        pass

    def get_method_name(self):
        """
        update the model parameters using the training parameters
        Returns: None

        """
        return 'DefaultViewportTimeModel'


class SimpleViewportTimeModel(DefaultViewportTimeModel):
    def __init__(self):
        super(SimpleViewportTimeModel, self).__init__()
        self.epsilon = 1e-3
        self.time_threshold = VPT_EPSILON

    def P(self, v, **kwargs):
        v = max(0., v)
        # C = 1 => P(v|C=1, E=1) = 1
        if 'C' in kwargs and kwargs['C'] == 1:
            return 1.0
        # S_e = 1 => vpt >= 0.3 s
        if 'S_e' in kwargs:
            if kwargs['S_e'] == 1:
                if v <= self.time_threshold:
                    return self.epsilon
                else:
                    return 1.0 # math.exp(-v * math.log(2.) / 50000.)
        # E = 1 => vpt > 0
        if 'E' in kwargs:
            if kwargs['E'] == 1 and v <= self.time_threshold:
                return self.epsilon
            else:
                return 1.0
        return 1.0

    def get_method_name(self):
        return 'SimpleViewportTimeModel'


class NormalViewportTimeModel(DefaultViewportTimeModel):
    def __init__(self, max_vertical_id):
        super(NormalViewportTimeModel, self).__init__()
        self.epsilon = 1e-6
        self.scale = 100000.
        self.time_threshold = self.convert_v(VPT_EPSILON)
        self.max_vertical_id = max_vertical_id

        self.s1 = [0.7 for _ in xrange(max_vertical_id)] # sigma
        self.m1 = [0.6 for _ in xrange(max_vertical_id)] # mu
        self.s0 = [0.06 for _ in xrange(max_vertical_id)]
        self.m0 = [0.01 for _ in xrange(max_vertical_id)]

        self.e0_p = [[] for _ in xrange(self.max_vertical_id)]
        self.e0_v = [[] for _ in xrange(self.max_vertical_id)]
        self.e1_p = [[] for _ in xrange(self.max_vertical_id)]
        self.e1_v = [[] for _ in xrange(self.max_vertical_id)]

    def get_method_name(self):
        return 'NormalViewportTimeModel'

    def normal_quality_function(self, x, s, m):
        return math.exp(-(x - m) ** 2 / (2 * s ** 2)) / (math.sqrt(2*math.pi) * s)

    def convert_v(self, v):
        v = math.log(max(self.epsilon, v/self.scale))
        return v

    def P(self, v, **kwargs):
        v = self.convert_v(v)
        layout = kwargs['layout']
        # C = 1 => P(v|C=1, E=1) = 1
        # E = 1 => vpt > 0
        if 'E' in kwargs:
            if (kwargs['E'] == 1) or ('C' in kwargs and kwargs['C'] == 1):
                if v >= self.time_threshold:
                    s = self.s1[layout]
                    m = self.m1[layout]
                    return max(self.normal_quality_function(v, s, m), self.epsilon)
                else:
                    return self.epsilon
            else:
                if v >= self.time_threshold:
                    s = self.s0[layout]
                    m = self.m0[layout]
                    return max(self.normal_quality_function(v, s, m), self.epsilon)
                else:
                    return 1.
        # return 1.0

    def update_init(self):
        """
        initialize the training parameters before M-step
        Returns: None

        """
        self.e0_p = [[] for _ in xrange(self.max_vertical_id)]
        self.e0_v = [[] for _ in xrange(self.max_vertical_id)]
        self.e1_p = [[] for _ in xrange(self.max_vertical_id)]
        self.e1_v = [[] for _ in xrange(self.max_vertical_id)]

    def update(self, posterior, v, **kwargs):
        """
        update the training parameters when there is an item in the Q-function of the form
        P(conditions|C_1..M, V_1..M) * log P(V_i|conditions)
        Args:
            posterior:  posterior prob P(conditions|C_1..M, V_1..M) estimated in E-step
            v: viewport time
            **kwargs: conditions su

        Returns: None

        """
        layout = kwargs['layout']
        v = self.convert_v(v)

        if ('C' in kwargs and kwargs['C'] == 1) or ('E' in kwargs and kwargs['E'] == 1):
            if v >= self.time_threshold:
                self.e1_p[layout].append(max(self.epsilon, posterior))
                self.e1_v[layout].append(v)
        else:
            if v >= self.time_threshold:
                self.e0_p[layout].append(max(self.epsilon, posterior))
                self.e0_v[layout].append(v)

    def updata_s_m(self, e_p, e_v, s_orgin, m_orgin):
        # get average k and t
        min_num = 50
        s_list, m_list = [], []
        avg_s, avg_m = [], []
        for i in xrange(len(e_p)):
            if len(e_p[i]) >= min_num:
                avg_s.append(s_orgin[i])
                avg_m.append(m_orgin[i])
        avg_s = np.mean(avg_s)
        avg_m = np.mean(avg_m)
        # update s, m
        for i in xrange(len(e_p)):
            if len(e_p[i]) < min_num:
                s_list.append(avg_s)
                m_list.append(avg_m)
                continue
            p = np.array(e_p[i], dtype=np.float64)
            v = np.array(e_v[i], dtype=np.float64)
            m = np.sum(p*v) / np.sum(p)
            s = math.sqrt(np.sum(p*np.power(v-m, 2))/np.sum(p))
            if math.fabs(s) < self.epsilon:
                s_list.append(self.epsilon)
                m_list.append(m)
                # print '~~~'
                continue
            s_list.append(s)
            m_list.append(m)
        # print s_list
        # print m_list
        # print ''
        return s_list, m_list

    def update_finalize(self):
        """
        update the model parameters using the training parameters
        Returns: None

        """
        self.s1, self.m1 = self.updata_s_m(self.e1_p, self.e1_v, self.s1, self.m1)
        self.s0, self.m0 = self.updata_s_m(self.e0_p, self.e0_v, self.s0, self.m0)

# class MixtureNormalViewportTimeModel(DefaultViewportTimeModel):
#     def __init__(self, max_vertical_id, num_dis_e1, num_dis_e0):
#         super(MixtureNormalViewportTimeModel, self).__init__()
#         self.epsilon = 1e-6
#         self.scale = 100000.
#         self.num_dis_e1 = num_dis_e1
#         self.num_dis_e0 = num_dis_e0
#
#         self.time_threshold = self.convert_v(VPT_EPSILON)
#         self.max_vertical_id = max_vertical_id
#
#         self.pi0 = [1. / self.num_dis_e0 for _ in range(self.num_dis_e0)]
#         self.pi1 = [1. / self.num_dis_e1 for _ in range(self.num_dis_e1)]
#
#         self.s0 = [] # sigma
#         self.m0 = [] # mu
#         self.s1 = []  # sigma
#         self.m1 = []  # mu
#         for i in range(self.max_vertical_id):
#             self.s0.append([0.7 for _ in xrange(self.num_dis_e0)])
#             self.m0.append([0.6 for _ in xrange(self.num_dis_e0)])
#             self.s1.append([0.7 for _ in xrange(self.num_dis_e1)])
#             self.m1.append([0.6 for _ in xrange(self.num_dis_e1)])
#
#         self.e0_p = [[] for _ in xrange(self.max_vertical_id)]
#         self.e0_v = [[] for _ in xrange(self.max_vertical_id)]
#         self.e1_p = [[] for _ in xrange(self.max_vertical_id)]
#         self.e1_v = [[] for _ in xrange(self.max_vertical_id)]
#
#     def normal_quality_function(self, x, s, m, pi):
#         prob = 0.
#         for i in range(len(pi)):
#             prob += pi[i] * math.exp(-(x - m[i]) ** 2 / (2 * s[i] ** 2)) / (math.sqrt(2*math.pi) * s[i])
#         return prob
#
#     def convert_v(self, v):
#         v = math.log(max(self.epsilon, v/self.scale))
#         return v
#
#     def P(self, v, **kwargs):
#         v = self.convert_v(v)
#         layout = kwargs['layout']
#         # C = 1 => P(v|C=1, E=1) = 1
#         # E = 1 => vpt > 0
#         if 'E' in kwargs:
#             if (kwargs['E'] == 1) or ('C' in kwargs and kwargs['C'] == 1):
#                 if v >= self.time_threshold:
#                     s = self.s1[layout]
#                     m = self.m1[layout]
#                     return max(self.normal_quality_function(v, s, m, self.pi1), self.epsilon)
#                 else:
#                     return self.epsilon
#             else:
#                 if v >= self.time_threshold:
#                     s = self.s0[layout]
#                     m = self.m0[layout]
#                     return max(self.normal_quality_function(v, s, m, self.pi0), self.epsilon)
#                 else:
#                     return 1.
#         # return 1.0
#
#     def update_init(self):
#         """
#         initialize the training parameters before M-step
#         Returns: None
#
#         """
#         self.e0_p = [[] for _ in xrange(self.max_vertical_id)]
#         self.e0_v = [[] for _ in xrange(self.max_vertical_id)]
#         self.e1_p = [[] for _ in xrange(self.max_vertical_id)]
#         self.e1_v = [[] for _ in xrange(self.max_vertical_id)]
#
#     def update(self, posterior, v, **kwargs):
#         """
#         update the training parameters when there is an item in the Q-function of the form
#         P(conditions|C_1..M, V_1..M) * log P(V_i|conditions)
#         Args:
#             posterior:  posterior prob P(conditions|C_1..M, V_1..M) estimated in E-step
#             v: viewport time
#             **kwargs: conditions su
#
#         Returns: None
#
#         """
#         layout = kwargs['layout']
#         v = self.convert_v(v)
#
#         if ('C' in kwargs and kwargs['C'] == 1) or ('E' in kwargs and kwargs['E'] == 1):
#             if v >= self.time_threshold:
#                 self.e1_p[layout].append(max(self.epsilon, posterior))
#                 self.e1_v[layout].append(v)
#         else:
#             if v >= self.time_threshold:
#                 self.e0_p[layout].append(max(self.epsilon, posterior))
#                 self.e0_v[layout].append(v)
#
#     def updata_s_m(self, e_p, e_v, s_orgin, m_orgin, pi_orgin):
#         # get average k and t
#         min_num = 50
#         s_list, m_list, pi_list = [], [], []
#         avg_s, avg_m, avg_pi = [], [], []
#         for i in xrange(len(e_p)):
#             if len(e_p[i]) >= min_num:
#                 avg_s.append(s_orgin[i])
#                 avg_m.append(m_orgin[i])
#         for i in xrange(len(pi_orgin)):
#             pi_orgin[i]
#         avg_s = np.mean(avg_s)
#         avg_m = np.mean(avg_m)
#         # update s, m
#         for i in xrange(len(e_p)):
#             if len(e_p[i]) < min_num:
#                 s_list.append(avg_s)
#                 m_list.append(avg_m)
#                 continue
#             p = np.array(e_p[i], dtype=np.float64)
#             v = np.array(e_v[i], dtype=np.float64)
#             m = np.sum(p*v) / np.sum(p)
#             s = math.sqrt(np.sum(p*np.power(v-m, 2))/np.sum(p))
#             if math.fabs(s) < self.epsilon:
#                 s_list.append(avg_s)
#                 m_list.append(avg_m)
#                 # print '~~~'
#                 continue
#             s_list.append(s)
#             m_list.append(m)
#         # print s_list
#         # print m_list
#         # print ''
#         return s_list, m_list
#
#     def update_finalize(self):
#         """
#         update the model parameters using the training parameters
#         Returns: None
#
#         """
#         self.s0, self.m0, self.pi0 = self.updata_s_m(self.e0_p, self.e0_v, self.s0, self.m0, self.pi0)
#         self.s1, self.m1, self.pi1 = self.updata_s_m(self.e1_p, self.e1_v, self.s1, self.m1, self.pi1)


class GammaViewportTimeModel(DefaultViewportTimeModel):
    def __init__(self, max_vertical_id):
        super(GammaViewportTimeModel, self).__init__()
        self.epsilon = 1e-6
        self.scale = 100000.
        self.time_threshold = self.convert_v(VPT_EPSILON)
        self.max_vertical_id = max_vertical_id
        self.k1 = [0.7 for _ in xrange(max_vertical_id)]
        self.t1 = [0.6 for _ in xrange(max_vertical_id)] # theta
        self.k0 = [0.06 for _ in xrange(max_vertical_id)]
        self.t0 = [0.01 for _ in xrange(max_vertical_id)]
        self.e0_p, self.e0_v, self.e1_p, self.e1_v = [], [], [], []

    def get_method_name(self):
        return 'GammaViewportTimeModel'

    def gamma_quality_function(self, x, k, t):
        return (x ** (k - 1.)) * math.exp(-x / t) / (math.gamma(k) * (t ** k))

    def convert_v(self, v):
        v = max(self.epsilon, v/self.scale)
        return v

    def P(self, v, **kwargs):
        v = self.convert_v(v)
        layout = kwargs['layout']
        # C = 1 => P(v|C=1, E=1) = 1
        # E = 1 => vpt > 0
        if 'E' in kwargs:
            if (kwargs['E'] == 1) or ('C' in kwargs and kwargs['C'] == 1):
                if v >= self.time_threshold:
                    k = self.k1[layout]
                    t = self.t1[layout]
                    return max(self.gamma_quality_function(v, k, t), self.epsilon)
                else:
                    return self.epsilon
            else:
                if v >= self.time_threshold:
                    k = self.k0[layout]
                    t = self.t0[layout]
                    return max(self.gamma_quality_function(v, k, t), self.epsilon)
                else:
                    return 1.
        # return 1.0

    def update_init(self):
        """
        initialize the training parameters before M-step
        Returns: None

        """
        self.e0_p = [[] for _ in xrange(self.max_vertical_id)]
        self.e0_v = [[] for _ in xrange(self.max_vertical_id)]
        self.e1_p = [[] for _ in xrange(self.max_vertical_id)]
        self.e1_v = [[] for _ in xrange(self.max_vertical_id)]

    def update(self, posterior, v, **kwargs):
        """
        update the training parameters when there is an item in the Q-function of the form
        P(conditions|C_1..M, V_1..M) * log P(V_i|conditions)
        Args:
            posterior:  posterior prob P(conditions|C_1..M, V_1..M) estimated in E-step
            v: viewport time
            **kwargs: conditions su

        Returns: None

        """
        layout = kwargs['layout']
        v = self.convert_v(v)

        if ('C' in kwargs and kwargs['C'] == 1) or ('E' in kwargs and kwargs['E'] == 1):
            if v >= self.time_threshold:
                self.e1_p[layout].append(max(self.epsilon, posterior))
                self.e1_v[layout].append(v)
        else:
            if v >= self.time_threshold:
                self.e0_p[layout].append(max(self.epsilon, posterior))
                self.e0_v[layout].append(v)

    def psi(self, x):
        delta = self.epsilon ** 2 * 1e-2
        y = math.gamma(x)
        y0 = math.gamma(x - delta)
        y1 = math.gamma(x + delta)
        return (y1-y0)/(2.*delta*y)

    def dpsi(self, x):
        delta = self.epsilon ** 2
        return (self.psi(x+delta) - self.psi(x-delta)) / (2. * delta)

    def updata_k_t(self, e_p, e_v, k_orgin, t_orgin):
        # get average k and t
        min_num = 50
        k_list, t_list = [], []
        avg_k, avg_t = [], []
        for i in xrange(len(e_p)):
            if len(e_p[i]) >= min_num:
                avg_k.append(k_orgin[i])
                avg_t.append(t_orgin[i])
        avg_k = np.mean(avg_k)
        avg_t = np.mean(avg_t)
        # update k, t
        for i in xrange(len(e_p)):
            if len(e_p[i]) < min_num:
                k_list.append(avg_k)
                t_list.append(avg_t)
                continue
            p = np.array(e_p[i], dtype=np.float64)
            v = np.array(e_v[i], dtype=np.float64)
            s = np.log(np.sum(p * v) / np.sum(p))
            s -= np.sum(p * np.log(v)) / np.sum(p)
            if math.fabs(s) < self.epsilon:
                k_list.append(k_orgin[i])
                t_list.append(t_orgin[i])
                # print '~~~'
                continue
            k = (3 - s + math.sqrt((s - 3.) ** 2 + 24 * s)) / (12 * s)
            t = 0.
            while True:
                if k < self.epsilon:
                    k = self.epsilon
                    t = np.sum(p * v) / (k * np.sum(p))
                    # print '!!!'
                    break
                if k > 50.:
                    k = avg_k
                    t = avg_t
                    # print '***'
                    break
                k_ = k
                k -= (math.log(k) - self.psi(k) - s) / (1. / k - self.dpsi(k))
                if math.fabs(k - k_) < self.epsilon:
                    t = np.sum(p * v) / (k * np.sum(p))
                    break
            k_list.append(k)
            t_list.append(t)
        # print k_list
        # print t_list
        # print ''
        return k_list, t_list

    def update_finalize(self):
        """
        update the model parameters using the training parameters
        Returns: None

        """
        self.k1, self.t1 = self.updata_k_t(self.e1_p, self.e1_v, self.k1, self.t1)
        self.k0, self.t0 = self.updata_k_t(self.e0_p, self.e0_v, self.k0, self.t0)


class WeibullViewportTimeModel(DefaultViewportTimeModel):
    def __init__(self, max_vertical_id):
        super(WeibullViewportTimeModel, self).__init__()
        self.epsilon = 1e-6
        self.scale = 10000.
        self.time_threshold = self.convert_v(VPT_EPSILON)
        self.max_vertical_id = max_vertical_id
        self.k1 = [1.2 for _ in xrange(max_vertical_id)]
        self.l1 = [0.6 for _ in xrange(max_vertical_id)] # lambda
        self.k0 = [1.2 for _ in xrange(max_vertical_id)]
        self.l0 = [0.6 for _ in xrange(max_vertical_id)]
        self.e0_p, self.e0_v, self.e1_p, self.e1_v = [], [], [], []

    def get_method_name(self):
        return 'WeibullViewportTimeModel'

    def weibull_quality_function(self, x, k, l):
        return k/l * (x/l)**(k-1) * math.exp(-(x/l)**k)

    def convert_v(self, v):
        v = max(self.epsilon, v/self.scale)
        return v

    def P(self, v, **kwargs):
        v = self.convert_v(v)
        layout = kwargs['layout']
        # C = 1 => P(v|C=1, E=1) = 1
        # E = 1 => vpt > 0
        if 'E' in kwargs:
            if (kwargs['E'] == 1) or ('C' in kwargs and kwargs['C'] == 1):
                if v >= self.time_threshold:
                    k = self.k1[layout]
                    l = self.l1[layout]
                    return max(self.weibull_quality_function(v, k, l), self.epsilon)
                else:
                    return self.epsilon
            else:
                if v >= self.time_threshold:
                    k = self.k0[layout]
                    l = self.l0[layout]
                    return max(self.weibull_quality_function(v, k, l), self.epsilon)
                else:
                    return 1.
        # return 1.0

    def update_init(self):
        """
        initialize the training parameters before M-step
        Returns: None

        """
        self.e0_p = [[] for _ in xrange(self.max_vertical_id)]
        self.e0_v = [[] for _ in xrange(self.max_vertical_id)]
        self.e1_p = [[] for _ in xrange(self.max_vertical_id)]
        self.e1_v = [[] for _ in xrange(self.max_vertical_id)]

    def update(self, posterior, v, **kwargs):
        """
        update the training parameters when there is an item in the Q-function of the form
        P(conditions|C_1..M, V_1..M) * log P(V_i|conditions)
        Args:
            posterior:  posterior prob P(conditions|C_1..M, V_1..M) estimated in E-step
            v: viewport time
            **kwargs: conditions su

        Returns: None

        """
        layout = kwargs['layout']
        v = self.convert_v(v)

        if ('C' in kwargs and kwargs['C'] == 1) or ('E' in kwargs and kwargs['E'] == 1):
            if v >= self.time_threshold:
                self.e1_p[layout].append(max(self.epsilon, posterior))
                self.e1_v[layout].append(v)
        else:
            if v >= self.time_threshold:
                self.e0_p[layout].append(max(self.epsilon, posterior))
                self.e0_v[layout].append(v)

    def get_l_from_k(self, p, v, k):
        return np.power(np.sum(p*np.power(v, k))/np.sum(p), 1./k)

    def get_k_from_k(self, p, v, k):
        l = self.get_l_from_k(p, v, k)
        k_ = math.log(l)
        k_ += np.sum(p*np.log(v/l)*np.power(v/l, k)) / np.sum(p)
        k_ -= np.sum(p*np.log(v)) / np.sum(p)
        k_ = 1. / k_
        return k_, l

    def updata_k_l(self, e_p, e_v, k_orgin, l_orgin):
        # get average k and t
        min_num = 50
        k_list, l_list = [], []
        avg_k, avg_l = [], []
        for i in xrange(len(e_p)):
            if len(e_p[i]) >= min_num:
                avg_k.append(k_orgin[i])
                avg_l.append(l_orgin[i])
        avg_k = np.median(avg_k)
        avg_l = np.median(avg_l)
        # update k, l
        for i in xrange(len(e_p)):
            if len(e_p[i]) < min_num:
                k_list.append(avg_k)
                l_list.append(avg_l)
                continue
            p = np.array(e_p[i], dtype=np.float64)
            v = np.array(e_v[i], dtype=np.float64)
            k = k_orgin[i]
            l = l_orgin[i]
            for _ in range(1):
                if k < self.epsilon:
                    k = self.epsilon
                    l = self.get_l_from_k(p, v, k)
                    # print '!!!'
                    break
                k_ = k
                k, l = self.get_k_from_k(p, v, k)
                if math.isnan(k) or math.isinf(l) or math.isnan(l) or math.isinf(k) or k > 50. or l > 50.:
                    k = avg_k
                    l = avg_l
                    # print '~~~'
                # if math.fabs(k - k_) < 10.: # self.epsilon
                #     break
            k_list.append(k)
            l_list.append(l)
        # print k_list
        # print l_list
        # print ''
        return k_list, l_list

    def update_finalize(self):
        """
        update the model parameters using the training parameters
        Returns: None

        """
        self.k1, self.l1 = self.updata_k_l(self.e1_p, self.e1_v, self.k1, self.l1)
        self.k0, self.l0 = self.updata_k_l(self.e0_p, self.e0_v, self.k0, self.l0)


class ComplexNormalViewportTimeModel(DefaultViewportTimeModel):
    def __init__(self, max_vertical_id):
        super(ComplexNormalViewportTimeModel, self).__init__()
        self.epsilon = 1e-6
        self.scale = 100000.
        self.time_threshold = self.convert_v(VPT_EPSILON)
        self.max_vertical_id = max_vertical_id

        self.s_e0 = [1.2 for _ in xrange(max_vertical_id)] # sigma
        self.m_e0 = [-1. for _ in xrange(max_vertical_id)] # mu
        self.s_e1_c = [1.2 for _ in xrange(max_vertical_id)]
        self.m_e1_c = [-1. for _ in xrange(max_vertical_id)]
        self.s_e1_se1 = [1.2 for _ in xrange(max_vertical_id)]
        self.m_e1_se1 = [-1. for _ in xrange(max_vertical_id)]
        self.s_e1_se0 = [1.2 for _ in xrange(max_vertical_id)]
        self.m_e1_se0 = [-1. for _ in xrange(max_vertical_id)]

        self.p_e0 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_e0 = [[] for _ in xrange(self.max_vertical_id)]
        self.p_e1_c = [[] for _ in xrange(self.max_vertical_id)]
        self.v_e1_c = [[] for _ in xrange(self.max_vertical_id)]
        self.p_e1_se1 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_e1_se1 = [[] for _ in xrange(self.max_vertical_id)]
        self.p_e1_se0 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_e1_se0 = [[] for _ in xrange(self.max_vertical_id)]

    def get_method_name(self):
        return 'ComplexNormalViewportTimeModel'

    def normal_quality_function(self, x, s, m):
        return math.exp(-(x - m) ** 2 / (2 * s ** 2)) / (math.sqrt(2*math.pi) * s)

    def convert_v(self, v):
        v = math.log(max(self.epsilon, v/self.scale))
        return v

    def P(self, v, **kwargs):
        v = self.convert_v(v)
        layout = kwargs['layout']
        if 'E' in kwargs and kwargs['E'] == 0:
            if 'C' in kwargs and kwargs['C'] == 1:
                return self.epsilon
            elif v < self.time_threshold:
                return 100.
            else:
                s = self.s_e0[layout]
                m = self.m_e0[layout]
                return max(self.normal_quality_function(v, s, m), self.epsilon)
        elif 'E' in kwargs and kwargs['E'] == 1:
            if v < self.time_threshold:
                return self.epsilon
            elif 'C' in kwargs and kwargs['C'] == 1:
                s = self.s_e1_c[layout]
                m = self.m_e1_c[layout]
                return max(self.normal_quality_function(v, s, m), self.epsilon)
            elif 'S_e' in kwargs and kwargs['S_e'] == 1:
                s = self.s_e1_se1[layout]
                m = self.m_e1_se1[layout]
                return max(self.normal_quality_function(v, s, m), self.epsilon)
            elif 'S_e' in kwargs and kwargs['S_e'] == 0:
                s = self.s_e1_se0[layout]
                m = self.m_e1_se0[layout]
                return max(self.normal_quality_function(v, s, m), self.epsilon)

    def update_init(self):
        """
        initialize the training parameters before M-step
        Returns: None

        """
        self.p_e0 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_e0 = [[] for _ in xrange(self.max_vertical_id)]
        self.p_e1_c = [[] for _ in xrange(self.max_vertical_id)]
        self.v_e1_c = [[] for _ in xrange(self.max_vertical_id)]
        self.p_e1_se1 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_e1_se1 = [[] for _ in xrange(self.max_vertical_id)]
        self.p_e1_se0 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_e1_se0 = [[] for _ in xrange(self.max_vertical_id)]

    def update(self, posterior, v, **kwargs):
        """
        update the training parameters when there is an item in the Q-function of the form
        P(conditions|C_1..M, V_1..M) * log P(V_i|conditions)
        Args:
            posterior:  posterior prob P(conditions|C_1..M, V_1..M) estimated in E-step
            v: viewport time
            **kwargs: conditions su

        Returns: None

        """
        v = self.convert_v(v)
        if v < self.time_threshold:
            return
        layout = kwargs['layout']
        posterior = max(self.epsilon, posterior)
        if 'E' in kwargs and kwargs['E'] == 0:
            if 'C' in kwargs and kwargs['C'] == 0:
                self.p_e0[layout].append(posterior)
                self.v_e0[layout].append(v)
        elif 'E' in kwargs and kwargs['E'] == 1:
            if 'C' in kwargs and kwargs['C'] == 1:
                self.p_e1_c[layout].append(posterior)
                self.v_e1_c[layout].append(v)
            elif 'S_e' in kwargs and kwargs['S_e'] == 1:
                self.p_e1_se1[layout].append(posterior)
                self.v_e1_se1[layout].append(v)
            elif 'S_e' in kwargs and kwargs['S_e'] == 0:
                self.p_e1_se0[layout].append(posterior)
                self.v_e1_se0[layout].append(v)

    def updata_s_m(self, e_p, e_v, s_orgin, m_orgin):
        # get average k and t
        min_num = 50
        s_list, m_list = [], []
        avg_s, avg_m = [], []
        for i in xrange(len(e_p)):
            if len(e_p[i]) >= min_num:
                avg_s.append(s_orgin[i])
                avg_m.append(m_orgin[i])
        avg_s = np.mean(avg_s)
        avg_m = np.mean(avg_m)
        # update s, m
        for i in xrange(len(e_p)):
            if len(e_p[i]) < min_num:
                s_list.append(avg_s)
                m_list.append(avg_m)
                continue
            p = np.array(e_p[i], dtype=np.float64)
            v = np.array(e_v[i], dtype=np.float64)
            m = np.sum(p*v) / np.sum(p)
            s = math.sqrt(np.sum(p*np.power(v-m, 2))/np.sum(p))
            if math.fabs(s) < self.epsilon:
                s_list.append(self.epsilon)
                m_list.append(m)
                # print '~~~'
                continue
            s_list.append(s)
            m_list.append(m)
        # print s_list
        # print m_list
        # print ''
        return s_list, m_list

    def update_finalize(self):
        """
        update the model parameters using the training parameters
        Returns: None

        """
        self.s_e0, self.m_e0 = self.updata_s_m(self.p_e0, self.v_e0, self.s_e0, self.m_e0)
        self.s_e1_c, self.m_e1_c = self.updata_s_m(self.p_e1_c, self.v_e1_c, self.s_e1_c, self.m_e1_c)
        self.s_e1_se1, self.m_e1_se1 = self.updata_s_m(self.p_e1_se1, self.v_e1_se1, self.s_e1_se1, self.m_e1_se1)
        self.s_e1_se0, self.m_e1_se0 = self.updata_s_m(self.p_e1_se0, self.v_e1_se0, self.s_e1_se0, self.m_e1_se0)


class ComplexGammaViewportTimeModel(DefaultViewportTimeModel):
    def __init__(self, max_vertical_id):
        super(ComplexGammaViewportTimeModel, self).__init__()
        self.epsilon = 1e-6
        self.scale = 100000.
        self.time_threshold = self.convert_v(VPT_EPSILON)
        self.max_vertical_id = max_vertical_id

        self.k_e0 = [0.7 for _ in xrange(max_vertical_id)] # k
        self.t_e0 = [0.6 for _ in xrange(max_vertical_id)] # theta
        self.k_e1_c = [0.7 for _ in xrange(max_vertical_id)]
        self.t_e1_c = [0.6 for _ in xrange(max_vertical_id)]
        self.k_e1_se1 = [0.7 for _ in xrange(max_vertical_id)]
        self.t_e1_se1 = [0.6 for _ in xrange(max_vertical_id)]
        self.k_e1_se0 = [0.7 for _ in xrange(max_vertical_id)]
        self.t_e1_se0 = [0.6 for _ in xrange(max_vertical_id)]

        self.p_e0 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_e0 = [[] for _ in xrange(self.max_vertical_id)]
        self.p_e1_c = [[] for _ in xrange(self.max_vertical_id)]
        self.v_e1_c = [[] for _ in xrange(self.max_vertical_id)]
        self.p_e1_se1 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_e1_se1 = [[] for _ in xrange(self.max_vertical_id)]
        self.p_e1_se0 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_e1_se0 = [[] for _ in xrange(self.max_vertical_id)]

    def get_method_name(self):
        return 'ComplexGammaViewportTimeModel'

    def gamma_quality_function(self, x, k, t):
        return (x ** (k - 1.)) * math.exp(-x / t) / (math.gamma(k) * (t ** k))

    def convert_v(self, v):
        v = max(self.epsilon, v/self.scale)
        return v

    def P(self, v, **kwargs):
        v = self.convert_v(v)
        layout = kwargs['layout']
        if 'E' in kwargs and kwargs['E'] == 0:
            if 'C' in kwargs and kwargs['C'] == 1:
                return self.epsilon
            elif v < self.time_threshold:
                return 100.
            else:
                k = self.k_e0[layout]
                l = self.t_e0[layout]
                return max(self.gamma_quality_function(v, k, l), self.epsilon)
        elif 'E' in kwargs and kwargs['E'] == 1:
            if v < self.time_threshold:
                return self.epsilon
            elif 'C' in kwargs and kwargs['C'] == 1:
                k = self.k_e1_c[layout]
                l = self.t_e1_c[layout]
                return max(self.gamma_quality_function(v, k, l), self.epsilon)
            elif 'S_e' in kwargs and kwargs['S_e'] == 1:
                k = self.k_e1_se1[layout]
                l = self.t_e1_se1[layout]
                return max(self.gamma_quality_function(v, k, l), self.epsilon)
            elif 'S_e' in kwargs and kwargs['S_e'] == 0:
                k = self.k_e1_se0[layout]
                l = self.t_e1_se0[layout]
                return max(self.gamma_quality_function(v, k, l), self.epsilon)

    def update_init(self):
        """
        initialize the training parameters before M-step
        Returns: None

        """
        self.p_e0 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_e0 = [[] for _ in xrange(self.max_vertical_id)]
        self.p_e1_c = [[] for _ in xrange(self.max_vertical_id)]
        self.v_e1_c = [[] for _ in xrange(self.max_vertical_id)]
        self.p_e1_se1 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_e1_se1 = [[] for _ in xrange(self.max_vertical_id)]
        self.p_e1_se0 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_e1_se0 = [[] for _ in xrange(self.max_vertical_id)]

    def update(self, posterior, v, **kwargs):
        """
        update the training parameters when there is an item in the Q-function of the form
        P(conditions|C_1..M, V_1..M) * log P(V_i|conditions)
        Args:
            posterior:  posterior prob P(conditions|C_1..M, V_1..M) estimated in E-step
            v: viewport time
            **kwargs: conditions su

        Returns: None

        """
        v = self.convert_v(v)
        if v < self.time_threshold:
            return
        layout = kwargs['layout']
        posterior = max(self.epsilon, posterior)
        if 'E' in kwargs and kwargs['E'] == 0:
            if 'C' in kwargs and kwargs['C'] == 0:
                self.p_e0[layout].append(posterior)
                self.v_e0[layout].append(v)
        elif 'E' in kwargs and kwargs['E'] == 1:
            if 'C' in kwargs and kwargs['C'] == 1:
                self.p_e1_c[layout].append(posterior)
                self.v_e1_c[layout].append(v)
            elif 'S_e' in kwargs and kwargs['S_e'] == 1:
                self.p_e1_se1[layout].append(posterior)
                self.v_e1_se1[layout].append(v)
            elif 'S_e' in kwargs and kwargs['S_e'] == 0:
                self.p_e1_se0[layout].append(posterior)
                self.v_e1_se0[layout].append(v)

    def psi(self, x):
        delta = self.epsilon ** 2 * 1e-2
        y = math.gamma(x)
        y0 = math.gamma(x - delta)
        y1 = math.gamma(x + delta)
        return (y1-y0)/(2.*delta*y)

    def dpsi(self, x):
        delta = self.epsilon ** 2
        return (self.psi(x+delta) - self.psi(x-delta)) / (2. * delta)

    def updata_k_t(self, e_p, e_v, k_orgin, t_orgin):
        # get average k and t
        min_num = 50
        k_list, t_list = [], []
        avg_k, avg_t = [], []
        for i in xrange(len(e_p)):
            if len(e_p[i]) >= min_num:
                avg_k.append(k_orgin[i])
                avg_t.append(t_orgin[i])
        avg_k = np.mean(avg_k)
        avg_t = np.mean(avg_t)
        # update k, t
        for i in xrange(len(e_p)):
            if len(e_p[i]) < min_num:
                k_list.append(avg_k)
                t_list.append(avg_t)
                continue
            p = np.array(e_p[i], dtype=np.float64)
            v = np.array(e_v[i], dtype=np.float64)
            s = np.log(np.sum(p * v) / np.sum(p))
            s -= np.sum(p * np.log(v)) / np.sum(p)
            if math.fabs(s) < self.epsilon:
                k_list.append(avg_k)
                t_list.append(avg_t)
                # print '~~~'
                continue
            k = (3 - s + math.sqrt((s - 3.) ** 2 + 24 * s)) / (12 * s)
            t = 0.
            while True:
                if k < self.epsilon:
                    k = self.epsilon
                    t = np.sum(p * v) / (k * np.sum(p))
                    # print '!!!'
                    break
                if k > 50.:
                    k = avg_k
                    t = avg_t
                    # print '***'
                    break
                k_ = k
                k -= (math.log(k) - self.psi(k) - s) / (1. / k - self.dpsi(k))
                if math.fabs(k - k_) < self.epsilon:
                    t = np.sum(p * v) / (k * np.sum(p))
                    break
            k_list.append(k)
            t_list.append(t)
        # print k_list
        # print t_list
        # print ''
        return k_list, t_list

    def update_finalize(self):
        """
        update the model parameters using the training parameters
        Returns: None

        """
        self.k_e0, self.t_e0 = self.updata_k_t(self.p_e0, self.v_e0, self.k_e0, self.t_e0)
        self.k_e1_c, self.t_e1_c = self.updata_k_t(self.p_e1_c, self.v_e1_c, self.k_e1_c, self.t_e1_c)
        self.k_e1_se1, self.t_e1_se1 = self.updata_k_t(self.p_e1_se1, self.v_e1_se1, self.k_e1_se1, self.t_e1_se1)
        self.k_e1_se0, self.t_e1_se0 = self.updata_k_t(self.p_e1_se0, self.v_e1_se0, self.k_e1_se0, self.t_e1_se0)


class ComplexWeibullViewportTimeModel(DefaultViewportTimeModel):
    def __init__(self, max_vertical_id):
        super(ComplexWeibullViewportTimeModel, self).__init__()
        self.epsilon = 1e-6
        self.scale = 1000.
        self.time_threshold = self.convert_v(VPT_EPSILON)
        self.max_vertical_id = max_vertical_id

        self.k_e0 = [1.2 for _ in xrange(max_vertical_id)] # k
        self.l_e0 = [0.6 for _ in xrange(max_vertical_id)] # lambda
        self.k_e1_c = [1.2 for _ in xrange(max_vertical_id)]
        self.l_e1_c = [0.6 for _ in xrange(max_vertical_id)]
        self.k_e1_se1 = [1.2 for _ in xrange(max_vertical_id)]
        self.l_e1_se1 = [0.6 for _ in xrange(max_vertical_id)]
        self.k_e1_se0 = [1.2 for _ in xrange(max_vertical_id)]
        self.l_e1_se0 = [0.6 for _ in xrange(max_vertical_id)]

        self.p_e0 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_e0 = [[] for _ in xrange(self.max_vertical_id)]
        self.p_e1_c = [[] for _ in xrange(self.max_vertical_id)]
        self.v_e1_c = [[] for _ in xrange(self.max_vertical_id)]
        self.p_e1_se1 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_e1_se1 = [[] for _ in xrange(self.max_vertical_id)]
        self.p_e1_se0 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_e1_se0 = [[] for _ in xrange(self.max_vertical_id)]

    def get_method_name(self):
        return 'ComplexWeibullViewportTimeModel'

    def weibull_quality_function(self, x, k, l):
        return k/l * (x/l)**(k-1) * math.exp(-(x/l)**k)

    def convert_v(self, v):
        v = max(self.epsilon, v/self.scale)
        return v

    def P(self, v, **kwargs):
        v = self.convert_v(v)
        layout = kwargs['layout']
        if 'E' in kwargs and kwargs['E'] == 0:
            if 'C' in kwargs and kwargs['C'] == 1:
                return self.epsilon
            elif v < self.time_threshold:
                return 100.
            else:
                k = self.k_e0[layout]
                l = self.l_e0[layout]
                return max(self.weibull_quality_function(v, k, l), self.epsilon)
        elif 'E' in kwargs and kwargs['E'] == 1:
            if v < self.time_threshold:
                return self.epsilon
            elif 'C' in kwargs and kwargs['C'] == 1:
                k = self.k_e1_c[layout]
                l = self.l_e1_c[layout]
                return max(self.weibull_quality_function(v, k, l), self.epsilon)
            elif 'S_e' in kwargs and kwargs['S_e'] == 1:
                k = self.k_e1_se1[layout]
                l = self.l_e1_se1[layout]
                return max(self.weibull_quality_function(v, k, l), self.epsilon)
            elif 'S_e' in kwargs and kwargs['S_e'] == 0:
                k = self.k_e1_se0[layout]
                l = self.l_e1_se0[layout]
                return max(self.weibull_quality_function(v, k, l), self.epsilon)

    def update_init(self):
        """
        initialize the training parameters before M-step
        Returns: None

        """
        self.p_e0 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_e0 = [[] for _ in xrange(self.max_vertical_id)]
        self.p_e1_c = [[] for _ in xrange(self.max_vertical_id)]
        self.v_e1_c = [[] for _ in xrange(self.max_vertical_id)]
        self.p_e1_se1 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_e1_se1 = [[] for _ in xrange(self.max_vertical_id)]
        self.p_e1_se0 = [[] for _ in xrange(self.max_vertical_id)]
        self.v_e1_se0 = [[] for _ in xrange(self.max_vertical_id)]

    def update(self, posterior, v, **kwargs):
        """
        update the training parameters when there is an item in the Q-function of the form
        P(conditions|C_1..M, V_1..M) * log P(V_i|conditions)
        Args:
            posterior:  posterior prob P(conditions|C_1..M, V_1..M) estimated in E-step
            v: viewport time
            **kwargs: conditions su

        Returns: None

        """
        v = self.convert_v(v)
        if v < self.time_threshold:
            return
        layout = kwargs['layout']
        posterior = max(self.epsilon, posterior)
        if 'E' in kwargs and kwargs['E'] == 0:
            if 'C' in kwargs and kwargs['C'] == 0:
                self.p_e0[layout].append(posterior)
                self.v_e0[layout].append(v)
        elif 'E' in kwargs and kwargs['E'] == 1:
            if 'C' in kwargs and kwargs['C'] == 1:
                self.p_e1_c[layout].append(posterior)
                self.v_e1_c[layout].append(v)
            elif 'S_e' in kwargs and kwargs['S_e'] == 1:
                self.p_e1_se1[layout].append(posterior)
                self.v_e1_se1[layout].append(v)
            elif 'S_e' in kwargs and kwargs['S_e'] == 0:
                self.p_e1_se0[layout].append(posterior)
                self.v_e1_se0[layout].append(v)

    def get_l_from_k(self, p, v, k):
        return np.power(np.sum(p*np.power(v, k))/np.sum(p), 1./k)

    def get_k_from_k(self, p, v, k):
        l = self.get_l_from_k(p, v, k)
        k_ = math.log(l)
        k_ += np.sum(p*np.log(v/l)*np.power(v/l, k)) / np.sum(p)
        k_ -= np.sum(p*np.log(v)) / np.sum(p)
        k_ = 1. / k_
        return k_, l

    def updata_k_l(self, e_p, e_v, k_orgin, l_orgin):
        # get average k and l
        min_num = 50
        k_list, l_list = [], []
        avg_k, avg_l = [], []
        for i in xrange(len(e_p)):
            if len(e_p[i]) >= min_num:
                avg_k.append(k_orgin[i])
                avg_l.append(l_orgin[i])
        avg_k = np.median(avg_k)
        avg_l = np.median(avg_l)
        # update k, l
        for i in xrange(len(e_p)):
            if len(e_p[i]) < min_num:
                k_list.append(avg_k)
                l_list.append(avg_l)
                continue
            p = np.array(e_p[i], dtype=np.float64)
            v = np.array(e_v[i], dtype=np.float64)
            k = k_orgin[i]
            l = l_orgin[i]
            for _ in range(10):
                if k < self.epsilon:
                    k = self.epsilon
                    l = self.get_l_from_k(p, v, k)
                    # print '!!!'
                    break
                k_ = k
                k, l = self.get_k_from_k(p, v, k)
                if math.isnan(k) or math.isinf(l) or math.isnan(l) or math.isinf(k) or k > 50. or l > 50.:
                    k = avg_k
                    l = avg_l
                    # print '~~~'
                # if math.fabs(k - k_) < 10.: # self.epsilon
                #     break
            k_list.append(k)
            l_list.append(l)
        # print k_list
        # print l_list
        # print ''
        return k_list, l_list

    def update_finalize(self):
        """
        update the model parameters using the training parameters
        Returns: None

        """
        self.k_e0, self.l_e0 = self.updata_k_l(self.p_e0, self.v_e0, self.k_e0, self.l_e0)
        self.k_e1_c, self.l_e1_c = self.updata_k_l(self.p_e1_c, self.v_e1_c, self.k_e1_c, self.l_e1_c)
        self.k_e1_se1, self.l_e1_se1 = self.updata_k_l(self.p_e1_se1, self.v_e1_se1, self.k_e1_se1, self.l_e1_se1)
        self.k_e1_se0, self.l_e1_se0 = self.updata_k_l(self.p_e1_se0, self.v_e1_se0, self.k_e1_se0, self.l_e1_se0)
