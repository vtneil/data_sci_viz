import sys


def main():
    argc, argv = len(sys.argv), sys.argv
    print('Hello world!')
    print(f'argc = {argc}')
    print(f'argv = {argv}')


if __name__ == '__main__':
    main()
