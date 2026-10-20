"""Small NERSC reader used by the Torch training stream."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_nersc_gauge(path: str | Path) -> tuple[np.ndarray, dict[str, str]]:
    """Read a 4D SU(3) 3x3 IEEE32 NERSC gauge file.

    The returned array has the same logical order as ``LatticeGauge.lexico``:
    ``(direction, t, z, y, x, row, column)``.
    """
    path = Path(path)
    header: dict[str, str] = {}
    with path.open("rb") as stream:
        if stream.readline() != b"BEGIN_HEADER\n":
            raise ValueError(f"not a NERSC gauge file: {path}")
        while True:
            line = stream.readline()
            if line == b"END_HEADER\n":
                offset = stream.tell()
                break
            key, value = line.decode("ascii").split("=", 1)
            header[key.strip()] = value.strip()
    if header.get("DATATYPE") != "4D_SU3_GAUGE_3x3":
        raise ValueError(f"unsupported NERSC datatype in {path}")
    dimensions = [int(header[f"DIMENSION_{index}"]) for index in range(1, 5)]
    if header.get("FLOATING_POINT") != "IEEE32LITTLE":
        raise ValueError("the training reader currently requires IEEE32LITTLE NERSC data")
    lx, ly, lz, lt = dimensions
    raw = np.fromfile(path, dtype="<c8", offset=offset)
    expected = lt * lz * ly * lx * 4 * 3 * 3
    if raw.size != expected:
        raise ValueError(f"unexpected data size in {path}: {raw.size} != {expected}")
    links = raw.reshape(lt, lz, ly, lx, 4, 3, 3)
    return np.ascontiguousarray(links.transpose(4, 0, 1, 2, 3, 5, 6)), header
