"""Block Python network I/O in WP00 checks, including inherited subprocesses."""
import os
import sys
from pathlib import Path

def audit(event, args):
    if event in {'socket.connect', 'socket.getaddrinfo', 'socket.sendto', 'socket.sendmsg'}:
        with Path(os.environ['WP00_NET_GUARD_LOG']).open('a') as stream:
            stream.write(f'blocked {event}\n')
        raise AssertionError('WP00 offline guard: network disabled')

sys.addaudithook(audit)
with Path(os.environ['WP00_NET_GUARD_LOG']).open('a') as stream:
    stream.write('guard-installed\n')
