import io
import subprocess
import typing as T
from pathlib import Path

PathLike = T.Union[str, Path]
PathOrBuffer = T.Union[PathLike, io.StringIO]


def run_subprocess_with_errorcheck(
    *popenargs,
    capture_output: bool = False,
    quiet: bool = False,
    env: dict[str, str] | None = None,
    shell: bool = False,
    executable: str | None = None,
    **kws,
) -> subprocess.CompletedProcess:
    """A command similar to subprocess.run, however the errormessage will
    contain the stderr when using this function. This makes it significantly
    easier to diagnose issues.
    """
    try:
        if capture_output:
            stdout = subprocess.PIPE
        elif quiet:
            stdout = subprocess.DEVNULL
        else:
            stdout = None

        p = subprocess.run(
            *popenargs,
            stderr=subprocess.PIPE,
            stdout=stdout,
            check=True,
            env=env,
            shell=shell,
            executable=executable,
            **kws,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Command failed with errorcode {e.returncode}.\n\n{e.stderr.decode()}"
        )
    return p
