"""Explicit Phase 8 commands for the current complete formal-data scope."""
import argparse
import json
import sys
from . import phase8_snapshot as s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('register-snapshot')
    for name in ('envelope','index','audit','pause','out'):
        p.add_argument('--'+name, required=True)
    p = commands.add_parser('summarize')
    p.add_argument('--registration', required=True);p.add_argument('--out', required=True)
    p = commands.add_parser('figures')
    p.add_argument('--summary', required=True);p.add_argument('--out', required=True)
    args = parser.parse_args()
    s.deny_network()
    try:
        if args.command == 'register-snapshot':
            result = s.register(args.envelope,args.index,args.audit,args.pause,args.out)
        elif args.command == 'summarize':
            result = s.execute(args.registration,args.out)
        else:
            from .figures import build_paper_figures
            result = build_paper_figures(args.summary,args.out)
        print(json.dumps({k:result[k] for k in ('status','kind','independent_tasks','trials') if k in result}))
        return 0
    except (ValueError,FileNotFoundError) as exc:
        print(json.dumps(dict(status='REFUSED',reason=str(exc))))
        return 3


if __name__ == '__main__':
    sys.exit(main())
