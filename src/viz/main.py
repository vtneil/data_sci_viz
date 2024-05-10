import sys
from viz.app import App


def main():
    argc, argv = len(sys.argv), sys.argv

    if argc < 2:
        path = 'src/viz/data/final_paper_format.json'
        print('Using example data.')
    else:
        path = argv[1]

    with App(path) as app:
        app.start()


if __name__ == '__main__':
    main()
