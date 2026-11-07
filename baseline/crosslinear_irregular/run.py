"""Run from repository root with python -m baseline.crosslinear_irregular.run."""
from baseline.common.run import make_parser, run


def main():
    run(make_parser('crosslinear').parse_args())


if __name__ == '__main__':
    main()
