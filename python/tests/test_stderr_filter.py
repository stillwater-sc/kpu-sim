"""
The `kpu` package filters NNPACK warnings out of stderr (#502).

The filter points fd 2 at a pipe. It used to read that pipe only at exit and
write what it read back to `sys.stderr`, which at that point was the pipe
itself, so everything written to stderr was lost: an uncaught exception's
traceback included (`examples/torch/model_compatibility.py` exited 1 with no
error text). A program writing more than the pipe buffer to stderr blocked
forever.

The filter is off under pytest, so each check runs in a child interpreter.
Runs under pytest, or as a script (exit 77 = skipped: `kpu` can't be imported).
"""

import os
import subprocess
import sys

PYTHON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(code, timeout=60):
    env = dict(os.environ)
    env["PYTHONPATH"] = PYTHON_DIR + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                          timeout=timeout)


def _kpu_importable():
    return _run("import kpu").returncode == 0


def test_traceback_reaches_stderr():
    r = _run("import kpu\nraise RuntimeError('visible traceback')")
    assert r.returncode == 1
    assert b"RuntimeError: visible traceback" in r.stderr


def test_lines_are_forwarded_and_nnpack_is_suppressed():
    r = _run("import kpu, os, sys\n"
             "print('python line', file=sys.stderr)\n"
             "os.write(2, b'raw line\\n[W NNPACK.cpp:64] Could not initialize NNPACK!\\n')\n")
    assert r.returncode == 0
    assert b"python line" in r.stderr
    assert b"raw line" in r.stderr
    assert b"NNPACK" not in r.stderr


def test_large_stderr_output_does_not_block():
    r = _run("import kpu, sys\nsys.stderr.write('x' * 1000000 + '\\n')\nprint('done')")
    assert r.returncode == 0
    assert b"done" in r.stdout
    assert r.stderr.count(b"x") == 1000000


if __name__ == "__main__":
    if not _kpu_importable():
        print("kpu is not importable here; skipping")
        sys.exit(77)
    tests = [test_traceback_reaches_stderr,
             test_lines_are_forwarded_and_nnpack_is_suppressed,
             test_large_stderr_output_does_not_block]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001 - report every failure
            failed += 1
            print(f"FAIL {t.__name__}: {e!r}")
    sys.exit(1 if failed else 0)
