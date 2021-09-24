#!/usr/bin/env python3

import colorama
import os
import subprocess
import sys
import threading

# dependency:
from mpi4py import MPI

class WrappedProcess:
    def __init__(self, argv, mpi_comm):
        self.mpi_comm = mpi_comm

        self.process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1,
            encoding="utf-8",
            shell=True,
            env=dict(os.environ,
                     # https://github.com/open-mpi/ompi/issues/6981
                     PMIX_MCA_gds="hash"))

        # Setup and start threads.
        self.stdin_thread = threading.Thread(
            target=self._root_stdin_thread if self.mpi_comm.Get_rank() == 0
                   else self._non_root_stdin_thread)
        self.stdin_thread.daemon = True
        self.stdin_thread.start()
        self.stdout_thread = threading.Thread(
            target=self._stdout_stderr_thread, args=(self.process.stdout,))
        self.stdout_thread.daemon = True
        self.stdout_thread.start()
        self.stderr_thread = threading.Thread(
            target=self._stdout_stderr_thread, args=(self.process.stderr,))
        self.stderr_thread.daemon = True
        self.stderr_thread.start()

    def _root_stdin_thread(self):
        for line in sys.stdin:
            self.mpi_comm.bcast(line, root=0)
            self.process.stdin.write(line)

    def _non_root_stdin_thread(self):
        while True:
            line = self.mpi_comm.bcast(None, root=0)
            self.process.stdin.write(line)

    def decorate_and_print(self, line):
        if line.endswith("\n"):
            line = line[:-1]
        # Prefix rank.
        line = f"{self.mpi_comm.Get_rank()}| {line}"
        if self.mpi_comm.Get_size() <= 4:
            # Colorize.
            color = {0: colorama.ansi.AnsiBack.RED,
                     1: colorama.ansi.AnsiBack.GREEN,
                     2: colorama.ansi.AnsiBack.BLUE,
                     3: colorama.ansi.AnsiBack.MAGENTA,
                }[self.mpi_comm.Get_rank()]
            line = f"\033[{color}m{line}\x1b[K\x1b[0m"
        print(line)

    def _stdout_stderr_thread(self, pipe):
        for line in pipe:
            self.decorate_and_print(line)

    def wait(self):
        self.process.wait()

def main():
    WrappedProcess(argv=sys.argv[1:], mpi_comm=MPI.COMM_WORLD).wait()

if __name__ == "__main__":
    main()
