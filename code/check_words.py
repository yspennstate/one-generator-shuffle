"""Check the explicit words of Lemmas 3.1 and 3.2.

For every odd n in the given range, build the words of the paper for c^i
(0 <= i < n) and for every transposition (i, i+d), evaluate them as
permutations of Z/nZ, and check that they are correct and no longer than
3h-2 and 12h-9, where h = ceil(log2 n).

Permutations are tuples p with p[x] the image of x. Products act on the
rightmost factor first: (p*q)[x] = p[q[x]].

Usage: python check_words.py [nmax]   (default nmax = 301)
"""
import sys


def compose(p, q):
    return tuple(p[x] for x in q)


def inverse(p):
    r = [0] * len(p)
    for x, y in enumerate(p):
        r[y] = x
    return tuple(r)


def generators(n):
    ident = tuple(range(n))
    c = tuple((x + 1) % n for x in range(n))
    tau = list(range(n))
    tau[0], tau[1] = 1, 0
    sigma = tuple((2 * x) % n for x in range(n))
    return {"e": ident, "c": c, "C": inverse(c), "t": tuple(tau),
            "s": sigma, "S": inverse(sigma)}


def evaluate(word, g, n):
    """Product of the letters of word, leftmost letter outermost."""
    p = g["e"]
    for letter in word:
        p = compose(p, g[letter])
    return p


def power_word(i):
    """Word for c^i from the recurrence c^(2q+b) = s c^q S c^b."""
    if i == 0:
        return ""
    if i == 1:
        return "c"
    q, b = divmod(i, 2)
    return "s" + power_word(q) + "S" + ("c" if b else "")


def inv_word(w):
    swap = {"c": "C", "C": "c", "s": "S", "S": "s", "t": "t"}
    return "".join(swap[x] for x in reversed(w))


def transposition_word(i, d):
    """Word for the transposition (i, i+d), 1 <= d < n."""
    bits = [j for j in range(d.bit_length()) if (d >> j) & 1]
    b = "s" * bits[0]
    for prev, nxt in zip(bits, bits[1:]):
        b += "tc" + "s" * (nxt - prev)
    core = b + "t" + inv_word(b)
    a = power_word(i)
    return a + core + inv_word(a)


def check(n):
    g = generators(n)
    h = (n - 1).bit_length()  # ceil(log2 n) for n not a power of two
    assert 2 ** (h - 1) < n <= 2 ** h
    sc = compose(compose(g["s"], g["c"]), g["S"])
    assert sc == compose(g["c"], g["c"]), "sigma c sigma^-1 != c^2"
    worst_power = 0
    for i in range(n):
        w = power_word(i)
        assert evaluate(w, g, n) == tuple((x + i) % n for x in range(n))
        assert len(w) <= (3 * h - 2 if i else 0)
        worst_power = max(worst_power, len(w))
    worst_t = 0
    for i in range(n):
        for d in range(1, n):
            w = transposition_word(i, d)
            p = evaluate(w, g, n)
            a, b = i, (i + d) % n
            target = list(range(n))
            target[a], target[b] = b, a
            assert p == tuple(target), (n, i, d)
            assert len(w) <= 12 * h - 9, (n, i, d, len(w))
            worst_t = max(worst_t, len(w))
    return h, worst_power, worst_t


def main():
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 301
    print("n h longest_power_word bound longest_transposition_word bound")
    for n in range(5, nmax + 1, 2):
        h, wp, wt = check(n)
        if n <= 31 or n % 50 == 1 or n == nmax:
            print(n, h, wp, 3 * h - 2, wt, 12 * h - 9)
    print("all odd n from 5 to", nmax, "pass")


if __name__ == "__main__":
    main()
