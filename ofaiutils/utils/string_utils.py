#!/usr/bin/env python

"""
Common string-related functionality
"""

def ensure_unicode(s):
    return s if type(s) == str else str(s,'utf8')

if __name__ == '__main__':
    pass
