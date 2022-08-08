def findAnagrams( s, p) :
    n = len(s)
    m = len(p)
    p_lis = [0] * 26
    res = []
    for i in range(m):
        p_lis[ord(p[i]) - ord('a')] += 1
    left, right = 0, m - 1
    while right < n:
        s_lis = [0] * 26
        for i in range(left, right):
            s_lis[ord(s[i]) - ord('a')] += 1
        if p_lis == s_lis:
            res.append(left)
        left += 1
        right += 1
    return res
s = "cbaebabacd"
p = "abc"
findAnagrams(s, p)