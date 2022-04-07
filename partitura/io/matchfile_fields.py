import numpy as np


class FractionalSymbolicDuration(object):
    """
    A class to represent symbolic duration information
    """

    def __init__(self, numerator, denominator=1, tuple_div=None, add_components=None):

        self.numerator = numerator
        self.denominator = denominator
        self.tuple_div = tuple_div
        self.add_components = add_components
        self.bound_integers(1024)

    def _str(self, numerator, denominator, tuple_div):
        if denominator == 1 and tuple_div is None:
            return str(numerator)
        else:
            if tuple_div is None:
                return "{0}/{1}".format(numerator, denominator)
            else:
                return "{0}/{1}/{2}".format(numerator, denominator, tuple_div)

    def bound_integers(self, bound):
        denominators = [
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            28,
            32,
            48,
            64,
            96,
            128,
        ]
        sign = np.sign(self.numerator) * np.sign(self.denominator)
        self.numerator = np.abs(self.numerator)
        self.denominator = np.abs(self.denominator)

        if self.numerator > bound or self.denominator > bound:
            val = float(self.numerator / self.denominator)
            dif = []
            for den in denominators:
                if np.round(val * den) > 0.9:
                    dif.append(np.abs(np.round(val * den) - val * den))
                else:
                    dif.append(np.abs(1 - val * den))

            difn = np.array(dif)
            min_idx = int(np.argmin(difn))

            self.denominator = denominators[min_idx]
            if int(np.round(val * self.denominator)) < 1:
                self.numerator = sign * 1
            else:
                self.numerator = sign * int(np.round(val * self.denominator))

    def __str__(self):

        if self.add_components is None:
            return self._str(self.numerator, self.denominator, self.tuple_div)
        else:
            r = [self._str(*i) for i in self.add_components]
            return "+".join(r)

    def __add__(self, sd):
        if isinstance(sd, int):
            sd = FractionalSymbolicDuration(sd, 1)

        dens = np.array([self.denominator, sd.denominator], dtype=int)
        new_den = np.lcm(dens[0], dens[1])
        a_mult = new_den // dens
        new_num = np.dot(a_mult, [self.numerator, sd.numerator])

        if self.add_components is None and sd.add_components is None:
            add_components = [
                (self.numerator, self.denominator, self.tuple_div),
                (sd.numerator, sd.denominator, sd.tuple_div),
            ]

        elif self.add_components is not None and sd.add_components is None:
            add_components = self.add_components + [
                (sd.numerator, sd.denominator, sd.tuple_div)
            ]
        elif self.add_components is None and sd.add_components is not None:
            add_components = [
                (self.numerator, self.denominator, self.tuple_div)
            ] + sd.add_components
        else:
            add_components = self.add_components + sd.add_components

        # Remove spurious components with 0 in the numerator
        add_components = [c for c in add_components if c[0] != 0]

        return FractionalSymbolicDuration(
            numerator=new_num, denominator=new_den, add_components=add_components
        )

    def __radd__(self, sd):
        return self.__add__(sd)

    def __float__(self):
        # Cast as float since the ability to return an instance of a strict
        # subclass of float is deprecated, and may be removed in a future
        # version of Python. (following a deprecation warning)
        return float(self.numerator / (self.denominator * (self.tuple_div or 1)))
