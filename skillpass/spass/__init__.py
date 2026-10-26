"""The S-Pass construction (Section 3).

`embed` turns a skill into the authorized model's token ids, embeddings e* and each token's
nearest-neighbour margin mu_i; `decoy` aligns the decoy document to the skill's length;
`objective` holds the hinge of Equation 4; `optimise` runs Adam on Equation 6 against the
frozen authorized model and its relatives; `make` ties them together for one skill and writes
the result; `file` is the on-disk format; `baselines` makes the `offset` and `decoy` tensors.
"""

from skillpass.spass.file import (
    PLAIN_FILE,
    SIDECAR_FILE,
    SPASS_FILE,
    SPass,
    read_plain,
    read_spass,
    write_plain,
    write_spass,
)

__all__ = [
    "PLAIN_FILE",
    "SIDECAR_FILE",
    "SPASS_FILE",
    "SPass",
    "read_plain",
    "read_spass",
    "write_plain",
    "write_spass",
]
