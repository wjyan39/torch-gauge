"""
SO(3) Clebsch-Gordan coefficients and the coupler
See Page 225, Eq. 3.8.49 of J. J. Sakurai, and Eq. 6, 9 of
 Schulten, Klaus, and Roy G. Gordon, Journal of Mathematical Physics 16.10 (1975): 1961-1970
"""


import os

import torch
from joblib import Memory
from sympy import N
from sympy.physics.quantum.cg import CG

from torch_gauge import ROOT_DIR
from torch_gauge.o3.spherical import SphericalTensor
from torch_gauge.o3.wigner import csh_to_rsh

memory = Memory(os.path.join(ROOT_DIR, ".o3_cache"), verbose=0)


class LeviCivitaCoupler(torch.nn.Module):
    """
    Simple tensor coupling module when max_l==1.
    The input spherical tensors must have n_rep_dims==1 and aligned dimensions
    """

    def __init__(self, metadata: torch.LongTensor):
        super().__init__()
        assert metadata.dim() == 1
        assert (
            len(metadata) == 2
        ), "Only SphericalTensor of max degree 1 is applicable for Cevi-Levita"
        self._metadata = metadata

    def forward(self, x1: SphericalTensor, x2: SphericalTensor, overlap_out=True):
        assert x1.metadata.shape[0] == 1
        assert x2.metadata.shape[0] == 1
        assert x1.rep_dims[0] == x2.rep_dims[0]
        coupling_dim = x1.rep_dims[0]
        assert torch.all(x1.metadata[0].eq(self._metadata))
        assert torch.all(x2.metadata[0].eq(self._metadata))
        ten_l1_1 = x1.ten.narrow(
            coupling_dim, self._metadata[0], self._metadata[1] * 3
        ).unflatten(coupling_dim, (3, self._metadata[1]))
        ten_l1_2 = x2.ten.narrow(
            coupling_dim, self._metadata[0], self._metadata[1] * 3
        ).unflatten(coupling_dim, (3, self._metadata[1]))
        # 0x0->0
        out_000 = x1.ten.narrow(coupling_dim, 0, self._metadata[0]) * x2.ten.narrow(
            coupling_dim, 0, self._metadata[0]
        )
        # 0x1->1
        out_011 = (
            x1.ten.narrow(coupling_dim, 0, self._metadata[1])
            .unsqueeze(coupling_dim)
            .mul(ten_l1_2)
        )
        # 1x0->1
        out_101 = (
            x2.ten.narrow(coupling_dim, 0, self._metadata[1])
            .unsqueeze(coupling_dim)
            .mul(ten_l1_1)
        )
        # 1x1->0
        out_110 = (ten_l1_1 * ten_l1_2).sum(coupling_dim)
        # 1x1->1, note that cross works since (y,z,x) is a canonical order
        out_111 = torch.cross(ten_l1_1, ten_l1_2, dim=coupling_dim)
        if overlap_out:
            # Align and contract the coupling outputs
            out_l0 = out_000
            out_l0.narrow(coupling_dim, 0, self._metadata[1]).add(out_110)
            out_l1 = (
                out_111.add(out_101)
                .add(out_011)
                .flatten(coupling_dim, coupling_dim + 1)
            )
            return x1.self_like(torch.cat([out_l0, out_l1], dim=coupling_dim))
        else:
            # Concatenate the coupling outputs to form a augmented tensor
            out_l0 = torch.cat([out_000, out_110], dim=coupling_dim)
            out_l1 = torch.cat(
                [out_101, out_011, out_111], dim=coupling_dim + 1
            ).flatten(coupling_dim, coupling_dim + 1)
            return SphericalTensor(
                torch.cat([out_l0, out_l1], dim=coupling_dim),
                rep_dims=(coupling_dim,),
                metadata=torch.LongTensor(
                    [[self._metadata[0] + self._metadata[1], self._metadata[1] * 3]]
                ),
            )


def get_clebsch_gordan_coefficient(j1, j2, j, m1, m2, m):
    """
    Generate Clebsch-Gordan coefficients using sympy with caching
    """
    # Matching the convention
    return float(N(CG(j1, m1, j2, m2, j, m).doit()))


# noinspection PyTypeChecker
@memory.cache
def get_rsh_cg_coefficients(j1, j2, j):
    csh_cg = torch.zeros(2 * j1 + 1, 2 * j2 + 1, 2 * j + 1, dtype=torch.double)
    for m1 in range(-j1, j1 + 1):
        for m2 in range(-j2, j2 + 1):
            if m1 + m2 < -j or m1 + m2 > j:
                continue
            csh_cg[j1 + m1, j2 + m2, j + m1 + m2] = get_clebsch_gordan_coefficient(
                j1, j2, j, m1, m2, m1 + m2
            )
    c2r_j1, c2r_j2, c2r_j = csh_to_rsh(j1), csh_to_rsh(j2), csh_to_rsh(j)
    # Adding a phase factor such that all coupling coefficients are real
    rsh_cg = torch.einsum(
        "abc,ai,bj,ck->ijk", csh_cg.to(torch.cdouble), c2r_j1, c2r_j2, c2r_j.conj()
    ) * (-1j) ** (j1 + j2 + j)
    assert torch.allclose(rsh_cg.imag, torch.zeros_like(csh_cg)), print(csh_cg, rsh_cg)
    return cg_compactify(rsh_cg.real, j1, j2, j)


def cg_compactify(coeffs, j1, j2, j):
    j1s = torch.arange(-j1, j1 + 1).view(2 * j1 + 1, 1, 1).expand_as(coeffs)
    j2s = torch.arange(-j2, j2 + 1).view(1, 2 * j2 + 1, 1).expand_as(coeffs)
    js = torch.arange(-j, j + 1).view(1, 1, 2 * j + 1).expand_as(coeffs)
    nonzero_mask = coeffs.abs() > 1e-12
    return torch.stack(
        [j1s[nonzero_mask], j2s[nonzero_mask], js[nonzero_mask], coeffs[nonzero_mask]],
        dim=0,
    )


class CGCoupler(torch.nn.Module):
    """General vectorized Clebsch-Gordan coupling module"""

    # TODO: Add doc

    def __init__(self, metadata: torch.LongTensor, overlap_out=True, trunc_in=False):
        super().__init__()
        assert metadata.dim == 1
        self._metadata = metadata
        self.layout = SphericalTensor.generate_rep_layout_1d_(self._metadata)
        if overlap_out:
            self.out_layout = self.layout
            self._init_params(overlap_out=overlap_out, trunc_in=trunc_in)
        else:
            raise NotImplementedError

    def _init_params(self, overlap_out, trunc_in):
        n_irreps_per_l = torch.arange(start=0, end=self._metadata.shape[0]) * 2 + 1
        repid_offsets = torch.cumsum(self._metadata * n_irreps_per_l, dim=0)
        repid_offsets = torch.cat([torch.LongTensor([0]), repid_offsets[:-1]])
        cg_tilde, repids_in1, repids_in2, repids_out = [], [], [], []
        if overlap_out:
            for lin1 in range(self._metadata.shape[0]):
                for lin2 in range(self._metadata.shape[0]):
                    for lout in range(abs(lin1 - lin2), lin1 + lin2 + 1):
                        if trunc_in:
                            # Only allows inputs that can saturate all output ls
                            if lin1+lin2 > self._metadata.shape[0]:
                                continue
                            degeneracy = min(
                                self._metadata[lin1],
                                self._metadata[lin2],
                                self._metadata[lin1 + lin2],
                            )
                        else:
                            if lout > self._metadata.shape[0]:
                                continue
                            degeneracy = min(
                                self._metadata[lin1],
                                self._metadata[lin2],
                                self._metadata[lout],
                            )
                        if degeneracy == 0:
                            continue
                        cg_source = get_rsh_cg_coefficients(lin1, lin2, lout)
                        cg_segment = cg_source.repeat_interleave(degeneracy, dim=1)
                        ns_segment = torch.arange(degeneracy).repeat(cg_source.shape[1])
                        # Calculating the representation IDs for the coupling tensors
                        repids_in1_3j = (
                            repid_offsets[lin1]
                            + (cg_segment[0] + lin1) * self._metadata[lin1]
                            + ns_segment
                        )
                        repids_in2_3j = (
                            repid_offsets[lin2]
                            + (cg_segment[1] + lin2) * self._metadata[lin2]
                            + ns_segment
                        )
                        repids_out_3j = (
                            repid_offsets[lout]
                            + (cg_segment[2] + lout) * self._metadata[lout]
                            + ns_segment
                        )
                        cg_tilde.append(cg_segment[3])
                        repids_in1.append(repids_in1_3j)
                        repids_in2.append(repids_in2_3j)
                        repids_out.append(repids_out_3j)
        else:
            raise NotImplementedError
        self.cg_tilde = torch.nn.Parameter(torch.cat(cg_tilde), requires_grad=False)
        self.repids_in1 = torch.nn.Parameter(torch.cat(repids_in1), requires_grad=False)
        self.repids_in2 = torch.nn.Parameter(torch.cat(repids_in2), requires_grad=False)
        self.repids_out = torch.nn.Parameter(torch.cat(repids_out), requires_grad=False)

    def forward(self, x1: SphericalTensor, x2: SphericalTensor) -> SphericalTensor:
        return NotImplementedError
