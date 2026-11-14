# fair distractors for numeric options aka T_opt

import bisect
import math
from collections import Counter


def chain_sums(x, w, gap, k):
    # x is the sorted log values and w the weights
    # for each m and j it sums the weight products over all sets of m values
    # that end at x j and have neighbours at least gap apart
    n = len(x)
    g = [None, list(w)]
    for m in range(2, k + 1):
        cur = [0.0] * n
        pre = 0.0
        i = 0
        for j in range(n):
            while x[i] + gap <= x[j] + 1e-9:
                pre += g[m - 1][i]
                i += 1
            cur[j] = w[j] * pre
        g.append(cur)
    return g


def rank_table(x, w, gap, k):
    # for every value as the answer gives the chance of each rank when k separated distractors come from w
    # also returns Z which is the chance the distractors are separated at all up to a constant
    n = len(x)
    lo = chain_sums(x, w, gap, k)
    # same thing from the top done on the mirrored list
    hi_rev = chain_sums([-v for v in reversed(x)], list(reversed(w)), gap, k)
    hi = [None] + [list(reversed(h)) for h in hi_rev[1:]]

    # prefix sums for the lower side and suffix sums for the upper side
    pre = [None] + [[0.0] * (n + 1) for _ in range(k)]
    suf = [None] + [[0.0] * (n + 1) for _ in range(k)]
    for m in range(1, k + 1):
        for j in range(n):
            pre[m][j + 1] = pre[m][j] + lo[m][j]
        for j in range(n - 1, -1, -1):
            suf[m][j] = suf[m][j + 1] + hi[m][j]

    Z = []
    ranks = []
    for t in range(n):
        a = bisect.bisect_right(x, x[t] - gap + 1e-9)   # values below that are far enough
        b = bisect.bisect_left(x, x[t] + gap - 1e-9)    # values above that are far enough
        L = [1.0] + [pre[m][a] for m in range(1, k + 1)]
        U = [1.0] + [suf[m][b] for m in range(1, k + 1)]
        # m distractors below the answer means rank m plus 1
        parts = [L[m] * U[k - m] for m in range(k + 1)]
        z = sum(parts)
        Z.append(z)
        ranks.append([p / z if z > 0 else 0.0 for p in parts])
    return Z, ranks


class FairSampler:
    def __init__(self, answers, s=0.2, k=3, shift=0.0, iters=1000, tol=1e-12):
        # answers is all answer strings of one task and k is the number of distractors
        # options have to be a ratio of 1 plus s apart which in log space is just a gap of log 1 plus s
        # shift is for road_length which has answers of 0 so there we use log of v plus 100
        vals = Counter()
        self.text = {}
        for a in answers:
            v = float(a)
            assert v + shift > 0
            vals[v] += 1
            self.text.setdefault(v, a)   # keep the original formatting like 2800 or 1900.0
        self.vals = sorted(vals)
        self.shift = shift
        self.x = [math.log(v + shift) for v in self.vals]
        n = sum(vals.values())
        self.p = [vals[v] / n for v in self.vals]
        self.gap = math.log(1 + s)
        self.k = k

        # drawing distractors straight from p and throwing away sets that are too close puts the
        # answer in the middle ranks too often and with s 0.2 rank 3 got up to 0.32 instead of 0.25 when i tried
        # drawing from q proportional to p over Z fixes it so every rank is equally likely
        # q is found by just iterating
        q = list(self.p)
        for it in range(iters):
            Z, _ = rank_table(self.x, q, self.gap, k)
            assert min(Z) > 0, "some answer cant get separated distractors, s too big?"
            new = [pv / z for pv, z in zip(self.p, Z)]
            tot = sum(new)
            new = [v / tot for v in new]
            diff = max(abs(a - b) for a, b in zip(new, q))
            q = new
            if diff < tol:
                break
        self.q = q
        self.iters = it + 1
        self.Z, self.ranks = rank_table(self.x, q, self.gap, k)

    def expected_ranks(self):
        # rank distribution of the answer when answers come from p which should be flat
        out = [0.0] * (self.k + 1)
        for pv, r in zip(self.p, self.ranks):
            for m in range(self.k + 1):
                out[m] += pv * r[m]
        return out

    def separated(self, xs):
        xs = sorted(xs)
        return all(xs[i + 1] - xs[i] >= self.gap - 1e-9 for i in range(len(xs) - 1))

    def draw(self, answer, rng, max_tries=100000):
        # returns k distractor strings for this answer
        y = math.log(float(answer) + self.shift)
        for _ in range(max_tries):
            idx = rng.choices(range(len(self.vals)), weights=self.q, k=self.k)
            if self.separated([y] + [self.x[i] for i in idx]):
                return [self.text[self.vals[i]] for i in idx]
        raise RuntimeError(f"couldnt find distractors for {answer}")
