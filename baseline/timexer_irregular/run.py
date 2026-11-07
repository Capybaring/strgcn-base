"""Run from repository root with python -m baseline.timexer_irregular.run."""
from baseline.common.run import make_parser, run


def main():
    run(make_parser('timexer').parse_args())


if __name__ == '__main__':
    main()
