class MulCompMap:
    def __init__(self, na: int, nb: int, sa: bool, sb: bool):
        self.na = na
        self.nb = nb
        self.sa = sa
        self.sb = sb

    def columns(self):
        return 1 if self.na == 1 and self.nb == 1 else self.nb + self.na - (not self.sb or self.sa)

    def shape(self):
        (na, nb, sa, sb) = (self.na, self.nb, self.sa, self.sb)

        res = []
        if na == 1 and nb == 1:
            res.append([7 if sa ^ sb else 8])
        else:
            col = 0

            # Crescending right triangle
            while col < nb:
                col += 1
                res.append([8] * col)
            # Central rectangle
            while col < na:
                col += 1
                res.append([8] * nb)
            # Decrescending left rectangle
            while col < nb + na - 1:
                col += 1
                res.append([8] * (nb + na - col))

            # Patch in sign handling
            if sa:
                for col in range(na - 1, na + nb - 1):
                    res[col][0] = ~res[col][0] & 15
            if sb:
                res[nb].insert(0, 2)
                for col in range(nb, nb + na - 1):
                    op = res[col][-1]
                    res[col][-1] = ((op & 3) << 2) | ((op >> 2) & 3)
                if not sa:
                    res.append([13])

        return res

    def absolute_term(self):
        (na, nb, sa, sb) = (self.na, self.nb, self.sa, self.sb)

        return (-1 if sa ^ sb else 0) if na == 1 and nb == 1 else ((-(sa | sb) << nb) | sa) << (na - 1)
