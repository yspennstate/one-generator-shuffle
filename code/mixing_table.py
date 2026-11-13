"""Exact total-variation mixing times for small odd n.

For each odd n in {5, 7, 9} the script propagates the full distribution of
the half-lazy walk on S_n started at the identity, for the base move set
{tau, c, c^-1} and for {tau, c, c^-1, sigma, sigma^-1} with sigma(x) = 2x mod n,
and reports the first t with ||P^t(e, .) - U||_TV <= eps for eps = 1/4 and
eps = 1/100. The walk holds with probability 1/2 and otherwise applies a move
chosen uniformly from the move set, by right multiplication g -> g s.

Usage: python mixing_table.py   (needs numpy; n = 9 uses about 40 MB)
"""
import itertools
import math

import numpy as np


def right_mult_index(perms, rank, s):
    """Index array j with perms[j[k]] = perms[k] o s."""
    return np.fromiter((rank[tuple(p[x] for x in s)] for p in perms),
                       dtype=np.int64, count=len(perms))


def moves(n, with_sigma):
    c = tuple((x + 1) % n for x in range(n))
    cinv = tuple((x - 1) % n for x in range(n))
    tau = list(range(n))
    tau[0], tau[1] = 1, 0
    out = [tuple(tau), c, cinv]
    if with_sigma:
        sigma = tuple((2 * x) % n for x in range(n))
        sinv = [0] * n
        for x, y in enumerate(sigma):
            sinv[y] = x
        out += [sigma, tuple(sinv)]
    assert len(set(out)) == len(out)
    return out


def mixing_times(n, with_sigma, eps_list, tmax=5000):
    perms = list(itertools.permutations(range(n)))
    rank = {p: k for k, p in enumerate(perms)}
    idx = [right_mult_index(perms, rank, s) for s in moves(n, with_sigma)]
    size = len(perms)
    u = 1.0 / size
    mu = np.zeros(size)
    mu[rank[tuple(range(n))]] = 1.0
    w = 0.5 / len(idx)
    found = {}
    for t in range(1, tmax + 1):
        new = 0.5 * mu
        for j in idx:
            new[j] += w * mu  # j is a permutation of the states
        mu = new
        tv = 0.5 * np.abs(mu - u).sum()
        for eps in eps_list:
            if eps not in found and tv <= eps:
                found[eps] = t
        if len(found) == len(eps_list):
            break
    return [found.get(eps) for eps in eps_list]


def best_added_generator(n, eps=0.25, tmax=200):
    """Smallest t(eps) over every added permutation pi of {0,...,n-1}.

    The move set is {tau, c, c^-1, pi, pi^-1} with duplicates and the identity
    removed; the walk holds with probability 1/2 and otherwise picks one of
    the distinct moves uniformly.
    """
    perms = list(itertools.permutations(range(n)))
    rank = {p: k for k, p in enumerate(perms)}
    ident = tuple(range(n))
    base = moves(n, False)
    cache = {}

    def index(s):
        if s not in cache:
            cache[s] = right_mult_index(perms, rank, s)
        return cache[s]

    best, winners = None, []
    for pi in perms:
        inv = [0] * n
        for x, y in enumerate(pi):
            inv[y] = x
        ms = []
        for s in base + [pi, tuple(inv)]:
            if s != ident and s not in ms:
                ms.append(s)
        idx = [index(s) for s in ms]
        mu = np.zeros(len(perms))
        mu[rank[ident]] = 1.0
        w = 0.5 / len(idx)
        t_hit = None
        for t in range(1, tmax + 1):
            new = 0.5 * mu
            for j in idx:
                new[j] += w * mu
            mu = new
            if 0.5 * np.abs(mu - 1.0 / len(perms)).sum() <= eps:
                t_hit = t
                break
        if t_hit is None:
            continue
        if best is None or t_hit < best:
            best, winners = t_hit, [pi]
        elif t_hit == best:
            winners.append(pi)
    return best, winners


def main():
    eps_list = [0.25, 0.01]
    print("n  h  base t(1/4)  with sigma t(1/4)  base t(1/100)  with sigma t(1/100)")
    for n in (5, 7, 9):
        b = mixing_times(n, False, eps_list)
        s = mixing_times(n, True, eps_list)
        h = math.ceil(math.log2(n))
        print(n, h, b[0], s[0], b[1], s[1])
    best, winners = best_added_generator(5)
    print("n = 5, best t(1/4) over all 120 added permutations:", best,
          "attained by", len(winners), "permutations, for example", winners[0])


if __name__ == "__main__":
    main()
