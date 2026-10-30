"""Canonical commands: python run.py two-body --show; python run.py hertz --show."""
import argparse,runpy,sys

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('two-body','two_body','hertz'))
    args=parser.parse_args(sys.argv[1:2]);remaining=sys.argv[2:]
    ##r## Independent physical verification command.
    sys.argv=[args.command]+remaining
    module='src.simulation.hertz' if args.command=='hertz' else 'src.simulation.compression'
    runpy.run_module(module,run_name='__main__')
if __name__=='__main__':main()
