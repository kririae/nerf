#!/usr/bin/env python3

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'torch using device: {device}')


def main():
    pass


if __name__ == '__main__':
    main()
