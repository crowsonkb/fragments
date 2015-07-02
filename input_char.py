#!/usr/bin/env python3

import copy
import sys
import tty

def input_char():
    """Read a single Unicode character from standard input."""
    fd = sys.stdin.fileno()
    old_mode = tty.tcgetattr(fd)
    try:
        # Modified from tty.setraw()
        mode = copy.deepcopy(old_mode)
        mode[tty.IFLAG] &= ~(tty.BRKINT | tty.INPCK | tty.ISTRIP | tty.IXON)
        mode[tty.CFLAG] &= ~(tty.CSIZE | tty.PARENB)
        mode[tty.CFLAG] |= tty.CS8
        mode[tty.LFLAG] &= ~(tty.ECHO | tty.ICANON | tty.IEXTEN)
        mode[tty.CC][tty.VMIN] = 1
        mode[tty.CC][tty.VTIME] = 0
        tty.tcsetattr(fd, tty.TCSADRAIN, mode)

        # Actually read the character
        char = sys.stdin.read(1)
    finally:
        tty.tcsetattr(fd, tty.TCSADRAIN, old_mode)
    return char

def main():
    print('Press a key...')
    char = input_char()
    print('I read:', repr(char))

if __name__ == '__main__':
    main()
