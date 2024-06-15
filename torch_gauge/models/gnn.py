"""
Native implementation of point-cloud based graph neural networks
 with torch-gauge functionalities
"""


import torch
from torch.nn import Linear, Parameter

from torch_gauge.geometric import poly_env
from torch_gauge.nn import SSP, IELin
from torch_gauge.o3.clebsch_gordan import LeviCivitaCoupler
from torch_gauge.o3.rsh import RSHxyz
from torch_gauge.o3.spherical import SphericalTensor
from torch_gauge.verlet_list import VerletList


class SchNetLayer(torch.nn.Module):
    """
    Schütt, Kristof, et al. "Schnet: A continuous-filter convolutional neural network for modeling quantum interactions."
     Advances in neural information processing systems. 2017.
    """

    def __init__(self, num_features):
        super().__init__()
        _nf = num_features
        self.gamma = 10.0
        self.rbf_centers = Parameter(
            torch.linspace(0.1, 30.1, 300), requires_grad=False
        )
        self.cfconv = torch.nn.Sequential(
            Linear(300, _nf, bias=False), SSP(), Linear(_nf, _nf, bias=False), SSP()
        )
        self.pre_conv = Linear(_nf, _nf)
        self.post_conv = torch.nn.Sequential(Linear(_nf, _nf), SSP(), Linear(_nf, _nf))

    def forward(self, vl: VerletList, l: int):
        xyz = vl.ndata["xyz"]
        pre_conv = self.pre_conv(vl.ndata[f"atomic_{l}"])
        d_ij = (vl.query_src(xyz) - xyz.unsqueeze(1)).norm(dim=2, keepdim=True)
        filters = self.cfconv(
            torch.exp(-self.gamma * (d_ij - self.rbf_centers.view(1, 1, -1)).pow(2))
        )
        conv_out = (filters * vl.query_src(pre_conv) * vl.edge_mask.unsqueeze(2)).sum(1)
        vl.ndata[f"atomic_{l + 1}"] = vl.ndata[f"atomic_{l}"] + self.post_conv(conv_out)
        return vl.ndata[f"atomic_{l + 1}"]


class SE3Layer(torch.nn.Module):
    """
    A generic rotational-equivariant layer restricted to l<=1, similar to:
    Thomas, Nathaniel, et al. "Tensor field networks: Rotation-and translation-equivariant neural networks for
     3d point clouds." arXiv preprint arXiv:1802.08219 (2018).
    """

    def __init__(self, num_features):
        super().__init__()
        _nf = num_features
        self.rbf_freqs = Parameter(torch.arange(16), requires_grad=False)
        self.rsh_mod = RSHxyz(max_l=1)
        self.coupler = LeviCivitaCoupler(torch.LongTensor([_nf, _nf]))
        self.filter_gen = torch.nn.Sequential(
            Linear(16, _nf, bias=False), SSP(), Linear(_nf, _nf, bias=False)
        )
        self.pre_conv = IELin([_nf, _nf], [_nf, _nf])
        self.post_conv = torch.nn.ModuleList(
            [
                IELin([2 * _nf, 3 * _nf], [_nf, _nf]),
                SSP(),
                IELin([_nf, _nf], [_nf, _nf]),
            ]
        )

    def forward(self, vl: VerletList, l: int):
        r_ij = vl.query_src(vl.ndata["xyz"]) - vl.ndata["xyz"].unsqueeze(1)
        d_ij = r_ij.norm(dim=2, keepdim=True)
        r_ij = r_ij / d_ij
        feat_in: SphericalTensor = vl.ndata[f"atomic_{l}"]
        pre_conv = vl.query_src(self.pre_conv(feat_in))
        filters_radial = self.filter_gen(
            torch.sin(d_ij / 5 * self.rbf_freqs.view(1, 1, -1))
            / d_ij
            * poly_env(d_ij / 5)
        )
        filters = pre_conv.self_like(
            self.rsh_mod(r_ij).ten.unsqueeze(-1).mul(filters_radial).flatten(2, 3)
        )
        coupling_out = self.post_conv[0](
            self.coupler(pre_conv, filters, overlap_out=False)
        )
        conv_out = feat_in.self_like(
            coupling_out.ten.mul(vl.edge_mask.unsqueeze(2)).sum(1)
        )
        conv_out = conv_out.scalar_mul(self.post_conv[1](conv_out.invariant()))
        conv_out.ten = self.post_conv[2](conv_out).ten
        vl.ndata[f"atomic_{l+1}"] = feat_in + conv_out
        return vl.ndata[f"atomic_{l+1}"]

from typing import Union
from torch_gauge.o3 import SphericalTensor, O3Tensor
from torch_gauge.o3.clebsch_gordan import CGCoupler, ModCGPCoupler, get_rsh_cg_coefficients_all
class Selfmix(torch.nn.Module):
    """
    Mixes features of different L orders within a given SphericalTensor.
    This is implemented according to that of PhiSNet, see 
    O.T.Unke, M.Gastegger, T.Smidt et. al. "SE(3)-equivariant prediction of molecular wavefunctions and electronic densities." 
    arXiv preprint arXiv:2106.02347 (2021).
    Also for this purpose, the original CGPCoupler is modified. 
    Args:
        metadata ::torch.LongTensor:: The metadata of the input SphericalTensor.
        order_out ::int:: The maximum output L order required during tensor product coupling.
        group ::string:: 'so3' for SphericalTensor, 'o3' for O3Tensor.
        cg_precision :: torch.tensor.type :: data type for cg-coupling.
    """
    def __init__(self, metadata: torch.LongTensor, group="so3", cg_precision=torch.float32):
        super().__init__()
        assert metadata.dim() == 1
        self.metadata_in = metadata 
        group = group.lower()
        if group == "so3":
            self.CGCoupler = CGCoupler(metadata, metadata, overlap_out=False, trunc_in=False, dtype=cg_precision)
            self.tensor_class = SphericalTensor
        elif group == "o3":
            self.CGCoupler = ModCGPCoupler(metadata, metadata, overlap_out=False, trunc_in=False, dtype=cg_precision)
            self.tensor_class = O3Tensor
        else:
            raise NotImplementedError
        self.group = group 
        num_channels_in = torch.sum(metadata).item() 
        num_channels_out = torch.sum(self.CGCoupler.metadata_out).item() 
        self.mix_coeff = torch.nn.Parameter(torch.Tensor(num_channels_out), requires_grad=True)
        self.keep_coeff = torch.nn.Parameter(torch.Tensor(num_channels_in), requires_grad=True)
        self._init_parameters()

    def _init_parameters(self):
        torch.nn.init.uniform_(self.keep_coeff, a=-torch.sqrt(3).item(), b=torch.sqrt(3).item())
        torch.nn.init.uniform_(self.mix_coeff, a=-torch.sqrt(3).item(), b=torch.sqrt(3).item())  
        # determine metadata_out 
        metadata_out = torch.zeros_like(self.CGCoupler.metadata_out)
        for (lpout, _, _, degeneracy) in self.CGCoupler.valid_coupling_ids:
            metadata_out[lpout] = max(metadata_out[lpout], degeneracy)
        # generate the flatenned idx for input, tensor product output and final coupled output
        tmp_zeros = torch.zeros(metadata_out.shape[0] - self.metadata_in.shape[0]).long()
        metadata_in = torch.cat([self.metadata_in, tmp_zeros], dim=0)
        metadata_tmp = torch.stack([metadata_in, metadata_out], dim=0)
        if self.group == "so3":
            stepwise = 1
            n_irreps_per_l =  torch.arange(start=0, end=metadata_out.shape[0]) * 2 + 1 
        elif self.group == "o3":
            stepwise = 2 
            n_irreps_per_l =  torch.arange(start=0, end=metadata_out.shape[0]//2) * 2 + 1
            n_irreps_per_l = n_irreps_per_l.repeat_interleave(2)
        else:
            raise NotImplementedError
        repid_offsets = torch.cumsum(
            metadata_tmp * n_irreps_per_l.unsqueeze(0), dim=1
        )
        repid_offsets = torch.cat(
            [torch.LongTensor([[0], [0]]), repid_offsets[:, :-1]], dim=1
        ).long() 
        # generate channels for tensor product output
        repids_tp_out = []
        tmp = [[] for _ in range(metadata_out.shape[0])]
        ## loop over each valid coupling (l1, l2, l) to decide the output index
        for (lpout, _, _, degeneracy) in self.CGCoupler.valid_coupling_ids:
            l_degeneracy = 2 * (lpout // stepwise) + 1
            ls_segement = torch.arange(l_degeneracy).repeat_interleave(degeneracy)
            ns_segement = torch.arange(degeneracy).repeat(l_degeneracy)
            repids_tp_out_3j = (
                repid_offsets[1, lpout]
                + ls_segement * metadata_out[lpout]
                + ns_segement
            ).view(l_degeneracy, -1)
            tmp[lpout].append(repids_tp_out_3j) 
        for tmp_list in tmp:
            if tmp_list:
                repids_tp_out.append(torch.cat(tmp_list, dim=1).view(-1))
        # generate channels for final sum up  
        metadata_cp = torch.minimum(self.metadata_in, metadata_out[:self.metadata_in.shape[0]])
        repids_in, repids_out = [], []
        for cur_lp in range(self.metadata_in.shape[0]):
            l_degeneracy = 2 * (cur_lp // stepwise) + 1
            ls_segement = torch.arange(l_degeneracy).repeat_interleave(metadata_cp[cur_lp]) 
            ns_segement = torch.arange(metadata_cp[cur_lp]).repeat(l_degeneracy) 
            repids_in_3j = (
                repid_offsets[0, cur_lp]
				+ ls_segement * metadata_tmp[0, cur_lp]
				+ ns_segement
            )
            repids_out_3j = (
                repid_offsets[1, cur_lp]
				+ ls_segement * metadata_tmp[1, cur_lp]
				+ ns_segement
			)
            repids_in.append(repids_in_3j)
            repids_out.append(repids_out_3j) 
        self.register_buffer("repids_in", torch.cat(repids_in).long())
        self.register_buffer("repids_out", torch.cat(repids_out).long())
        self.register_buffer("repids_tp_out", torch.cat(repids_tp_out).long())
        self.register_buffer("out_layout", self.tensor_class.generate_rep_layout_1d_(metadata_out))
        self.metadata_out = metadata_out
        self.num_channels = torch.sum(metadata_out).item()

    def forward(self, x:Union[SphericalTensor, O3Tensor]) -> Union[SphericalTensor, O3Tensor]:
        """
        Rules:
            k_{l_3} \odot x^{(l_3)} + \sum_{l_1} \sum_{l_2} s_{l_3, l_2, l_1} \odot (x^{(l_1)} \ocross x^{(l_2)})
        """
        assert len(x.rep_dims) == 1
        assert torch.all(x.metadata[0].eq(self.metadata_in)).item()
        coupling_dim = x.rep_dims[0]
        
        # k_{l_3} \odot x^{(l_3)}, as (1)
        tmp_x = x 
        broadcast_shape_k = tuple(
            self.keep_coeff.shape[0] if d == coupling_dim else 1 for d in range(x.ten.dim())
        )
        tmp_x = tmp_x.scalar_mul(self.keep_coeff.view(broadcast_shape_k))
        
        # s_{l_3, l_2, l_1} \odot (x^{(l_1)} \ocross x^{(l_2)}) as (2)
        cat_tp_out = self.CGCoupler(x, x) 
        cat_tp_out.ten = cat_tp_out.ten * 0.5
        broadcast_shape_m = tuple(
            self.mix_coeff.shape[0] if d == coupling_dim else 1 for d in range(x.ten.dim())
        )
        cat_tp_out = cat_tp_out.scalar_mul(self.mix_coeff.view(broadcast_shape_m))
        # \sum_{l_1} \sum_{l_2} {(2)} as (3)
        out_shape = tuple(
            self.out_layout.shape[1] if d == coupling_dim else x.ten.shape[d] for d in range(x.ten.dim())
        )
        out_tp_ten = torch.zeros(
            out_shape, dtype=x.ten.dtype, device=x.ten.device
        ).index_add_(
            coupling_dim, self.repids_tp_out, cat_tp_out.ten
        )
        # (1) + (3)
        out_tp_ten.index_add_(
            coupling_dim,
            self.repids_out,
            tmp_x.ten.index_select(dim=coupling_dim, index=self.repids_in)
        )

        return self.tensor_class(
            data_ten=out_tp_ten,
            rep_dims=(coupling_dim,),
            metadata=self.metadata_out.unsqueeze(0),
            num_channels=(self.num_channels,),
            rep_layout=(self.out_layout,),
        )


class SphLinear(torch.nn.Module):
    """
    Spherical Linear layer in PhiSNet.
    """
    def __init__(self, metadata_in: torch.LongTensor, target_metadata: torch.LongTensor, group="so3", cg_precision=torch.float32):
        assert metadata_in.dim() == 1
        assert target_metadata.dim() == 1 
        super().__init__()
        self.metadata_in = metadata_in
        self.metadata_out = target_metadata
        self.group = group
        self.SelfMix = Selfmix(metadata=metadata_in, group=group, cg_precision=cg_precision) 
        inter_metadata = self.SelfMix.metadata_out 
        self.IELinear = IELin(metadata_in=inter_metadata, metadata_out=target_metadata, group=group)

    def forward(self, x: Union[SphericalTensor, O3Tensor]) -> Union[SphericalTensor, O3Tensor]:
        assert torch.all(x.metadata[-1].eq(self.metadata_in)).item()
        mix_x = self.SelfMix(x)
        return self.IELinear(mix_x)


class TPExpansion(torch.nn.Module):
    """
    Tensor Production Expansion Layer.
    Inverse to tensor product contractions, used to expand SphericalTensor 
    irreps to (2l_1 + 1) x (2l_2 + 1) matrix that represents its contribution 
    to the direct sum representation of the tensor product of two irreps of 
    degree l1 and l2. 
    Args:
        metadata ::torch.LongTensor:: metadata of input SphericalTensor.
        l1 ::int:: output angular momentum of bra-tensor.
        l2 ::int:: output angular momentum of ket-tensor.
        overlap_out ::bool:: if True, the output tensor is one matrix, else several (2l_1 + 1) x (2l_2 + 1) matrices.
    """
    def __init__(self, 
        metadata:torch.LongTensor, 
        l1:int, 
        l2:int, 
        overlap_out=True, 
        dtype=torch.double
    ):
        assert metadata.dim() == 1
        super().__init__()
        self.max_l = metadata.shape[0] - 1 
        self.l1 = l1
        self.l2 = l2 
        self.metadata = metadata
        self.dtype = dtype
        self._init_params(overlap_out)
    
    def _init_params(self, overlap_out:bool):
        n_irreps_per_l = torch.arange(start=0, end=self.max_l+1) * 2 + 1
        
        repid_offsets_in = torch.cumsum(
            self.metadata * n_irreps_per_l, dim=0
        )
        repid_offsets_in = torch.cat(
            [torch.LongTensor([0]), repid_offsets_in[:-1]], dim=0
        ).long() 

        degeneracy = self.metadata.min().item()
        if degeneracy == 0:
            degeneracy = 1 
        cg_tilde, repids_in, repids_out = [], [], [] 
        block_dim = (2*self.l1 + 1) * (2*self.l2 + 1)
        # loop over l
        for l in range(self.max_l + 1):
            if self.metadata[l] == 0: continue
            if self.l1 + self.l2 < l or abs(self.l1 - self.l2) > l:
                continue 
            # l --> l1 \osum l2
            cg_source = get_rsh_cg_coefficients_all(self.l1, self.l2, l) 
            cg_segment = cg_source.repeat_interleave(degeneracy, dim=1)
            ns_segment = torch.arange(degeneracy).repeat(cg_source.shape[1]) 
            repids_in_3j = (
                repid_offsets_in[l] 
                + (cg_segment[2] + l) * self.metadata[l]
                + ns_segment
            )
            
            if overlap_out:
                repids_out_blk = torch.arange(block_dim).repeat_interleave((2*l+1)*degeneracy)
            else:
                single_ton = torch.arange(block_dim).repeat_interleave(2*l+1).view(block_dim, -1)
                repids_out_blk = torch.cat([single_ton + i_offset * block_dim for i_offset in range(degeneracy)], dim=1).flatten()

            cg_tilde.append(cg_segment[3])
            repids_in.append(repids_in_3j) 
            repids_out.append(repids_out_blk)

        self.reshaping = True if not overlap_out else False
        self.out_feat_dim = block_dim * degeneracy
        self.register_buffer("cg_tilde", torch.cat(cg_tilde).type(self.dtype))
        self.register_buffer("repids_in", torch.cat(repids_in).long())
        self.register_buffer("repids_out", torch.cat(repids_out).long())

    def forward(self, x:SphericalTensor) -> torch.Tensor:
        assert len(x.rep_dims) == 1
        assert torch.all(x.metadata[0].eq(self.metadata))
        coupling_dim = x.rep_dims[0]
        x_tilde = torch.index_select(x.ten, dim=coupling_dim, index=self.repids_in) 
        broadcast_shape = tuple(
            self.cg_tilde.shape[0] if d == coupling_dim else 1 for d in range(x.ten.dim())
        )
        tp_tilde = x_tilde * self.cg_tilde.view(broadcast_shape) 
        out_shape = tuple(
            self.out_feat_dim if d == coupling_dim else x.ten.shape[d] for d in range(x.ten.dim())
        )
        out_ten:torch.Tensor = torch.zeros(
            out_shape, dtype=x_tilde.dtype, device=x_tilde.device 
        ).index_add_(
            coupling_dim, self.repids_out, tp_tilde 
        )

        if self.reshaping:
            block_dim = (2*self.l1 + 1) * (2*self.l2 + 1)
            out_ten = out_ten.view(*x.ten.shape[:coupling_dim], -1, block_dim, *x.ten.shape[coupling_dim+1:])
        return out_ten
    

class TPExpansionO3(torch.nn.Module):
    """
    Parity awared Tensor Production Expansion Layer.
    Inverse to tensor product contractions, used to expand O3Tensor 
    irreps to (2l_1 + 1) x (2l_2 + 1) matrix that represents its contribution 
    to the direct sum representation of the tensor product of two irreps of 
    degree and parity (l1, p1) and (l2, p2). 
    Args:
        metadata ::torch.LongTensor:: metadata of input SphericalTensor.
        l1 ::int:: output angular momentum of bra-tensor.
        p1 ::int:: output parity of bra-tensor.
        l2 ::int:: output angular momentum of ket-tensor.
        p2 ::int:: output parity of ket-tensor.
        overlap_out ::bool:: if True, the output tensor is one matrix, else several (2l_1 + 1) x (2l_2 + 1) matrices.
    """
    def __init__(self, 
        metadata:torch.LongTensor, 
        l1:int,
        p1:int, 
        l2:int,
        p2:int, 
        overlap_out=True, 
        dtype=torch.float32
    ):
        assert metadata.dim() == 1
        # assert metadata.minimum().item() > 0
        super().__init__()
        self.max_l = metadata.shape[0] // 2 - 1
        self.l1, self.p1 = l1, p1
        self.l2, self.p2 = l2, p2  
        self.metadata = metadata
        self.dtype = dtype
        self._init_params(overlap_out)
    
    def _init_params(self, overlap_out:bool):
        n_irreps_per_l = torch.arange(start=0, end=self.max_l+1) * 2 + 1
        n_irreps_per_l = n_irreps_per_l.repeat_interleave(2)
        
        repid_offsets_in = torch.cumsum(
            self.metadata * n_irreps_per_l, dim=0
        )
        repid_offsets_in = torch.cat(
            [torch.LongTensor([0]), repid_offsets_in[:-1]], dim=0
        ).long() 

        non_zero_mask = self.metadata > 0
        degeneracy = self.metadata[non_zero_mask].min().item()

        cg_tilde, repids_in, repids_out = [], [], [] 
        block_dim = (2*self.l1 + 1) * (2*self.l2 + 1)
        # loop over l
        for lin in range(self.max_l + 1):
            for pin in (1, -1):
                # angular selection rule 
                if self.l1 + self.l2 < lin or abs(self.l1 - self.l2) > lin:
                    continue
                coupling_parity = (-1)**(self.l1 + self.l2 + lin) 
                # parity selection rule
                if self.p1 * self.p2 * coupling_parity != pin:
                    continue 
                lpin = 2 * lin + (1 - pin) // 2        
                if self.metadata[lpin] == 0: continue

                # l --> l1 \osum l2
                cg_source = get_rsh_cg_coefficients_all(self.l1, self.l2, lin) 
                cg_segment = cg_source.repeat_interleave(degeneracy, dim=1)
                ns_segment = torch.arange(degeneracy).repeat(cg_source.shape[1]) 
                repids_in_3j = (
                    repid_offsets_in[lpin] 
                    + (cg_segment[2] + lin) * self.metadata[lpin]
                    + ns_segment
                )
                if overlap_out:
                    repids_out_blk = torch.arange(block_dim).repeat_interleave((2*lin+1)*degeneracy)
                else:
                    single_ton = torch.arange(block_dim).repeat_interleave(2*lin+1).view(block_dim, -1)
                    repids_out_blk = torch.cat([single_ton + i_offset * block_dim for i_offset in range(degeneracy)], dim=1).flatten()

                cg_tilde.append(cg_segment[3])
                repids_in.append(repids_in_3j) 
                repids_out.append(repids_out_blk)

        self.reshaping = True if not overlap_out else False
        self.out_feat_dim = block_dim * degeneracy
        self.register_buffer("cg_tilde", torch.cat(cg_tilde).type(self.dtype))
        self.register_buffer("repids_in", torch.cat(repids_in).long())
        self.register_buffer("repids_out", torch.cat(repids_out).long())

    def forward(self, x:O3Tensor) -> torch.Tensor:
        assert len(x.rep_dims) == 1
        assert torch.all(x.metadata[0].eq(self.metadata))
        coupling_dim = x.rep_dims[0]
        x_tilde = torch.index_select(x.ten, dim=coupling_dim, index=self.repids_in) 
        broadcast_shape = tuple(
            self.cg_tilde.shape[0] if d == coupling_dim else 1 for d in range(x.ten.dim())
        )
        tp_tilde = x_tilde * self.cg_tilde.view(broadcast_shape) 
        
        out_shape = tuple(
            self.out_feat_dim if d == coupling_dim else x.ten.shape[d] for d in range(x.ten.dim())
        )
        out_ten:torch.Tensor = torch.zeros(
            out_shape, dtype=x_tilde.dtype, device=x_tilde.device 
        ).index_add_(
            coupling_dim, self.repids_out, tp_tilde 
        )
        if self.reshaping:
            block_dim = (2*self.l1 + 1) * (2*self.l2 + 1)
            out_ten = out_ten.view(*x.ten.shape[:coupling_dim], -1, block_dim, *x.ten.shape[coupling_dim+1:])
        return out_ten
