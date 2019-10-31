#!/usr/bin/env python

def parse_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--file', type=str, default='a')

def main():
    args = parse_args()

if __name__ == '__main__':
    main()