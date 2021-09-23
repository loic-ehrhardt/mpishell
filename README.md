# mpishell

A small program for interactive sessions with MPI applications.

## Usage

`mpishell` redirects the standard input to all the jobs and decorates their standard output to make it easier to identify the job emitting them.

Consider the following example:

```
$ mpirun -np 2 cat
hello
hello
```

Here, `hello` was written to standard input, `mpirun` redirected it to the first job, running `cat`, which echoed the same word back. The second MPI job did nothing.

Now if one interleaves `mpishell` between `mpirun` and `cat`:

```
$ mpirun -np 2 mpishell cat
hello
0| hello
1| hello
```

This time, `mpishell` forwarded `hello` to the two jobs, and both reply the same word. `mpishell` also prepends the line with the job rank. Note that `mpishell` also colorizes the lines if the number of jobs is low.

`mpishell` can then be used with more complex programs, like `bash` or `ipython`:

```
$ mpirun -np 2 mpishell ipython3
0| Python 3.8.10 (default, Jun  2 2021, 10:49:15)
0| Type 'copyright', 'credits' or 'license' for more information
0| IPython 7.13.0 -- An enhanced Interactive Python. Type '?' for help.
0| /usr/lib/python3/dist-packages/IPython/core/interactiveshell.py:935: UserWarning: Attempting to work in a virtualenv. If you encounter problems, please install IPython inside the virtualenv.
0|   warn("Attempting to work in a virtualenv. If you encounter problems, please "
0|
1| Python 3.8.10 (default, Jun  2 2021, 10:49:15)
1| Type 'copyright', 'credits' or 'license' for more information
1| IPython 7.13.0 -- An enhanced Interactive Python. Type '?' for help.
1| /usr/lib/python3/dist-packages/IPython/core/interactiveshell.py:935: UserWarning: Attempting to work in a virtualenv. If you encounter problems, please install IPython inside the virtualenv.
1|   warn("Attempting to work in a virtualenv. If you encounter problems, please "
1|
from mpi4py import MPI
0| In [1]:
1| In [1]:
MPI.COMM_WORLD.Get_rank()
0| In [2]: Out[2]: 0
0|
1| In [2]: Out[2]: 1
1|
```
