import json
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conf', action='append')
    return parser


def parse_args(parser, conf):
    lst_args = []
    if conf is not None:
        for name_conf in conf:
            with open(name_conf, 'rt') as f:
                t_args = argparse.Namespace()
                t_args.__dict__.update(json.load(f))
                args = parser.parse_args(namespace=t_args)
                lst_args.append(args)
    return lst_args
