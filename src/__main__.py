import argparse,runpy,sys

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('two-body','two_body','hertz','crossed-cylinders','cylinder-validation'))
    args=parser.parse_args(sys.argv[1:2]);remaining=sys.argv[2:]
    sys.argv=[args.command]+remaining
    module={'cylinder-validation':'src.simulation.test_cylinder_validation','hertz':'src.simulation.hertz','crossed-cylinders':'src.simulation.test_crossed_cylinders'}.get(args.command,'src.simulation.compression')  # ##r##
    runpy.run_module(module,run_name='__main__')
if __name__=='__main__':main()
