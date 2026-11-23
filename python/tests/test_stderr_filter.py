"""
The `kpu` package filters NNPACK warnings out of stderr (#502).

The filter points fd 2 at a pipe. It used to read that pipe only at exit and
write what it read back to `sys.stderr`, which at that point was the pipe
itself, so everything written to stderr was lost: an uncaught exception's
traceback included (`examples/torch/model_compatibility.py` exited 1 with no
error text). A program writing more than the pipe buffer to stderr blocked
forever.

The filter is off under pytest, so each check runs in a child interpreter.
Runs under pytest, or as a script (exit 77 = skipped: a third-party module `kpu`
needs is missing; any other import failure fails).
"""

import os
import re
import subprocess
import sys

PYTHON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(code, timeout=60):
    env = dict(os.environ)
    env["PYTHONPATH"] = PYTHON_DIR + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                          timeout=timeout)


def _missing_dependency():
    """Name of the missing third-party module that keeps `kpu` from importing, or None.

    Any other import failure is a regression, so it raises with the child's output.
    """
    r = _run("import kpu")
    if r.returncode == 0:
        return None
    m = re.search(rb"ModuleNotFoundError: No module named '([\w.]+)'", r.stderr)
    if m and not m.group(1).startswith(b"kpu"):
        return m.group(1).decode()
    raise RuntimeError("import kpu failed:\n" + r.stderr.decode(errors="replace"))


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


def test_carriage_return_records_are_forwarded_and_filtered():
    r = _run("import kpu, os\n"
             "os.write(2, b'step 1\\rstep 2\\r[W NNPACK.cpp:64] NNPACK\\rstep 3')\n")
    assert r.returncode == 0
    assert r.stderr == b"step 1\rstep 2\rstep 3"


def test_unterminated_output_is_forwarded_while_running():
    # 200 KB with no newline, then the child blocks on stdin. The parent must
    # receive more than the 64 KB bound while the child is still running.
    env = dict(os.environ)
    env["PYTHONPATH"] = PYTHON_DIR + os.pathsep + env.get("PYTHONPATH", "")
    child = subprocess.Popen(
        [sys.executable, "-c", "import kpu, os, sys\nos.write(2, b'y' * 200000)\nsys.stdin.read()"],
        env=env, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    import threading
    received = []
    threshold = threading.Event()

    def drain():
        # One reader drains to EOF; the event marks the 64 KB threshold.
        total = 0
        while True:
            chunk = child.stderr.read1(65536)
            if not chunk:
                return
            received.append(chunk)
            total += len(chunk)
            if total > 65536:
                threshold.set()

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    try:
        threshold.wait(timeout=30)
        # Snapshot before releasing the child: at its exit the rest arrives anyway.
        received_while_running = sum(len(c) for c in list(received))
        still_running = child.poll() is None
        child.stdin.close()
        reader.join(timeout=30)
        child.wait(timeout=30)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    assert still_running
    assert received_while_running > 65536


def test_overlong_record_is_forwarded_whole_even_if_it_mentions_nnpack():
    r = _run("import kpu, os\nos.write(2, b'NNPACK ' + b'z' * 200000)\n")
    assert r.returncode == 0
    assert r.stderr.count(b"z") == 200000


def test_nnpack_after_the_overflow_boundary_is_still_forwarded():
    # One record: 128 KB, then "NNPACK tail", then its newline. Every fragment of
    # it is forwarded; suppression resumes with the next record.
    r = _run("import kpu, os\n"
             "os.write(2, b'a' * 131072 + b' NNPACK tail\\n')\n"
             "os.write(2, b'[W NNPACK.cpp:64] Could not initialize NNPACK!\\nafter\\n')\n")
    assert r.returncode == 0
    assert r.stderr == b"a" * 131072 + b" NNPACK tail\nafter\n"


def test_large_stderr_output_does_not_block():
    r = _run("import kpu, sys\nsys.stderr.write('x' * 1000000 + '\\n')\nprint('done')")
    assert r.returncode == 0
    assert b"done" in r.stdout
    assert r.stderr.count(b"x") == 1000000


if __name__ == "__main__":
    missing = _missing_dependency()
    if missing:
        print(f"kpu needs {missing}, which isn't installed here; skipping")
        sys.exit(77)
    tests = [test_traceback_reaches_stderr,
             test_lines_are_forwarded_and_nnpack_is_suppressed,
             test_carriage_return_records_are_forwarded_and_filtered,
             test_unterminated_output_is_forwarded_while_running,
             test_overlong_record_is_forwarded_whole_even_if_it_mentions_nnpack,
             test_nnpack_after_the_overflow_boundary_is_still_forwarded,
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
