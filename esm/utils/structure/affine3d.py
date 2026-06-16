from __future__ import annotations

import typing as T
from abc import ABC
from dataclasses import dataclass

import torch
from torch.nn import functional as F
from typing_extensions import Self

from esm.utils.misc import fp32_autocast_context


class Rotation(ABC):
    @classmethod
    def identity(
        cls, shape: tuple[int, ...], **tensor_kwargs
    ) -> Self: ...  # ty:ignore[empty-body]

    @classmethod
    def random(
        cls, shape: tuple[int, ...], **tensor_kwargs
    ) -> Self: ...  # ty:ignore[empty-body]

    def __getitem__(self, idx: T.Any) -> Self: ...  # ty:ignore[empty-body]

    @property
    def tensor(self) -> torch.Tensor:  # ty:ignore[empty-body]
        # We claim that this should be zero-cost abstraction that returns the raw tensor backing this
        # object. The raw tensor should always have exactly 1 more dim than self.shape, which should be
        # implemented using reshaping
        ...

    @property
    def shape(self) -> torch.Size:  # ty:ignore[empty-body]
        # The "shape" of the rotation, as if it was a torch.tensor object
        # This means that 1x4 quaternions are treated as size (1,) for example
        ...

    def as_matrix(self) -> RotationMatrix: ...  # ty:ignore[empty-body]

    def as_quat(
        self, normalize: bool = False
    ) -> RotationQuat: ...  # ty:ignore[empty-body]

    def compose(self, other: Self) -> Self:  # ty:ignore[empty-body]
        # To be safe, we force users to explicitly convert between rotation types.
        ...

    def convert_compose(self, other: Self) -> Self:  # ty:ignore[empty-body]
        # This function will automatically convert between types of rotations
        ...

    def apply(self, p: torch.Tensor) -> torch.Tensor:  # ty:ignore[empty-body]
        # rotates points by this rotation object
        ...

    def invert(self) -> Self: ...  # ty:ignore[empty-body]

    @property
    def dtype(self) -> torch.dtype:
        return self.tensor.dtype

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    @property
    def requires_grad(self) -> bool:
        return self.tensor.requires_grad

    @classmethod
    def _from_tensor(cls, t: torch.Tensor) -> Self:
        # This function exists to simplify the below functions, esp type signatures
        # Its implementation is different from Affine3D.from_tensor and does not
        # autodetect rotation types.
        return cls(t)  # type: ignore

    def to(self, **kwargs) -> Self:
        return self._from_tensor(self.tensor.to(**kwargs))

    def detach(self, *args, **kwargs) -> Self:
        return self._from_tensor(self.tensor.detach(**kwargs))

    def tensor_apply(self, func) -> Self:
        # Applys a function to the underlying tensor
        return self._from_tensor(
            torch.stack([func(x) for x in self.tensor.unbind(dim=-1)], dim=-1)
        )


class RotationMatrix(Rotation):
    def __init__(self, rots: torch.Tensor):
        if rots.shape[-1] == 9:
            rots = rots.unflatten(-1, (3, 3))
        assert rots.shape[-1] == 3
        assert rots.shape[-2] == 3
        # Force full precision
        rots = rots.to(torch.float32)
        self._rots = rots

    @classmethod
    def identity(cls, shape, **tensor_kwargs):
        rots = torch.eye(3, **tensor_kwargs)
        rots = rots.view(*[1 for _ in range(len(shape))], 3, 3)
        rots = rots.expand(*shape, -1, -1)
        return cls(rots)

    @classmethod
    def random(cls, shape, **tensor_kwargs):
        return RotationQuat.random(shape, **tensor_kwargs).as_matrix()

    def __getitem__(self, idx: T.Any) -> RotationMatrix:
        indices = (idx,) if isinstance(idx, int) or idx is None else tuple(idx)
        return RotationMatrix(self._rots[indices + (slice(None), slice(None))])

    @property
    def shape(self) -> torch.Size:
        return self._rots.shape[:-2]

    def as_matrix(self) -> RotationMatrix:
        return self

    def as_quat(self, normalize: bool = False) -> RotationQuat:
        m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
            self._rots.flatten(-2), dim=-1
        )
        q_abs = _sqrt_subgradient(
            torch.stack(
                [
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ],
                dim=-1,
            )
        )
        # we produce the desired quaternion multiplied by each of r, i, j, k
        quat_by_rijk = torch.stack(
            [
                x
                for lst in [
                    [q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01],
                    [m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20],
                    [m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21],
                    [m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2],
                ]
                for x in lst
            ],
            dim=-1,
        ).unflatten(-1, (4, 4))

        # We floor here at 0.1 but the exact level is not important; if q_abs is small,
        # the candidate won't be picked.
        flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
        quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

        # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
        # forall i; we pick the best-conditioned one (with the largest denominator)
        # We manually implement one_hot so torch.compile works
        one_hot = torch.zeros_like(q_abs, dtype=torch.bool)
        one_hot.scatter_(-1, q_abs.argmax(dim=-1, keepdim=True), True)
        quat = quat_candidates[one_hot, :].reshape(q_abs.shape)
        return RotationQuat(quat)

    def compose(self, other: RotationMatrix) -> RotationMatrix:
        with fp32_autocast_context(self._rots.device.type):
            return RotationMatrix(self._rots @ other._rots)

    def convert_compose(self, other: Rotation):
        return self.compose(other.as_matrix())

    def apply(self, p: torch.Tensor) -> torch.Tensor:
        with fp32_autocast_context(self.device.type):
            if self._rots.shape[-3] == 1:
                # This is a slight speedup over einsum for batched rotations
                return p @ self._rots.transpose(-1, -2).squeeze(-3)
            else:
                # einsum way faster than bmm!
                return torch.einsum("...ij,...j", self._rots, p)

    def invert(self) -> RotationMatrix:
        return RotationMatrix(self._rots.transpose(-1, -2))

    @property
    def tensor(self) -> torch.Tensor:
        return self._rots.flatten(-2)

    def to_3x3(self) -> torch.Tensor:
        return self._rots

    @staticmethod
    def from_graham_schmidt(
        x_axis: torch.Tensor, xy_plane: torch.Tensor, eps: float = 1e-12
    ) -> RotationMatrix:
        # A low eps here is necessary for good stability!
        return RotationMatrix(_graham_schmidt(x_axis, xy_plane, eps))


class RotationQuat(Rotation):
    def __init__(self, quats: torch.Tensor, normalized=False):
        assert quats.shape[-1] == 4
        self._normalized = normalized
        # Force float32 as well
        if normalized:
            self._quats = F.normalize(quats.to(torch.float32), dim=-1)
            self._quats = self._quats.where(self._quats[..., :1] >= 0, -self._quats)
        else:
            self._quats = quats.to(torch.float32)

    @classmethod
    def identity(cls, shape, **tensor_kwargs):
        q = torch.ones((*shape, 4), **tensor_kwargs)
        mult = torch.tensor([1, 0, 0, 0], device=q.device)
        return RotationQuat(q * mult)

    @classmethod
    def random(cls, shape, **tensor_kwargs):
        quat = torch.randn((*shape, 4), **tensor_kwargs)
        return RotationQuat(quat, normalized=True)

    def __getitem__(self, idx: T.Any) -> RotationQuat:
        indices = (idx,) if isinstance(idx, int) or idx is None else tuple(idx)
        return RotationQuat(self._quats[indices + (slice(None),)])

    @property
    def shape(self) -> torch.Size:
        return self._quats.shape[:-1]

    def compose(self, other: RotationQuat) -> RotationQuat:
        with fp32_autocast_context(self._quats.device.type):
            return RotationQuat(_quat_mult(self._quats, other._quats))

    def convert_compose(self, other: Rotation):
        return self.compose(other.as_quat())

    def as_matrix(self) -> RotationMatrix:
        q = self.normalized().tensor
        r, i, j, k = torch.unbind(q, -1)
        two_s = 2.0 / torch.linalg.norm(q, dim=-1)

        o = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            -1,
        )
        return RotationMatrix(o.reshape(q.shape[:-1] + (3, 3)))

    def as_quat(self, normalize: bool = False) -> RotationQuat:
        return self

    def apply(self, p: torch.Tensor) -> torch.Tensor:
        return _quat_rotation(self.normalized()._quats, p)

    def invert(self) -> RotationQuat:
        return RotationQuat(_quat_invert(self._quats))

    @property
    def tensor(self) -> torch.Tensor:
        return self._quats

    def normalized(self) -> RotationQuat:
        return self if self._normalized else RotationQuat(self._quats, normalized=True)


@dataclass(frozen=True)
class Affine3D:
    trans: torch.Tensor
    rot: Rotation

    def __post_init__(self):
        assert self.trans.shape[:-1] == self.rot.shape

    @staticmethod
    def identity(
        shape_or_affine: T.Union[tuple[int, ...], "Affine3D"],
        rotation_type: T.Type[Rotation] = RotationMatrix,
        **tensor_kwargs,
    ):
        # Creates a new identity Affine3D object with a specified shape
        # or the same shape as another Affine3D object.
        if isinstance(shape_or_affine, Affine3D):
            kwargs = {"dtype": shape_or_affine.dtype, "device": shape_or_affine.device}
            kwargs.update(tensor_kwargs)
            shape = shape_or_affine.shape
            rotation_type = type(shape_or_affine.rot)
        else:
            kwargs = tensor_kwargs
            shape = shape_or_affine
        return Affine3D(
            torch.zeros((*shape, 3), **kwargs),  # ty:ignore[no-matching-overload]
            rotation_type.identity(shape, **kwargs),
        )

    @staticmethod
    def random(
        shape: tuple[int, ...],
        std: float = 1,
        rotation_type: T.Type[Rotation] = RotationMatrix,
        **tensor_kwargs,
    ) -> "Affine3D":
        return Affine3D(
            trans=torch.randn((*shape, 3), **tensor_kwargs).mul(std),
            rot=rotation_type.random(shape, **tensor_kwargs),
        )

    def __getitem__(self, idx: T.Any) -> "Affine3D":
        indices = (idx,) if isinstance(idx, int) or idx is None else tuple(idx)
        return Affine3D(trans=self.trans[indices + (slice(None),)], rot=self.rot[idx])

    @property
    def shape(self) -> torch.Size:
        return self.trans.shape[:-1]

    @property
    def dtype(self) -> torch.dtype:
        return self.trans.dtype

    @property
    def device(self) -> torch.device:
        return self.trans.device

    @property
    def requires_grad(self) -> bool:
        return self.trans.requires_grad

    def to(self, **kwargs) -> "Affine3D":
        return Affine3D(self.trans.to(**kwargs), self.rot.to(**kwargs))

    def detach(self, *args, **kwargs) -> "Affine3D":
        return Affine3D(self.trans.detach(**kwargs), self.rot.detach(**kwargs))

    def tensor_apply(self, func) -> "Affine3D":
        # Applys a function to the underlying tensor
        return self.from_tensor(
            torch.stack([func(x) for x in self.tensor.unbind(dim=-1)], dim=-1)
        )

    def as_matrix(self):
        return Affine3D(trans=self.trans, rot=self.rot.as_matrix())

    def as_quat(self, normalize: bool = False):
        return Affine3D(trans=self.trans, rot=self.rot.as_quat(normalize))

    def compose(self, other: "Affine3D", autoconvert: bool = False):
        rot = self.rot
        new_rot = (rot.convert_compose if autoconvert else rot.compose)(other.rot)
        new_trans = rot.apply(other.trans) + self.trans
        return Affine3D(trans=new_trans, rot=new_rot)

    def compose_rotation(self, other: Rotation, autoconvert: bool = False):
        return Affine3D(
            trans=self.trans,
            rot=(self.rot.convert_compose if autoconvert else self.rot.compose)(other),
        )

    def scale(self, v: torch.Tensor | float):
        return Affine3D(self.trans * v, self.rot)

    def mask(self, mask: torch.Tensor, with_zero=False):
        # Returns a transform where True positions in mask is identity
        if with_zero:
            tensor = self.tensor
            return Affine3D.from_tensor(
                torch.zeros_like(tensor).where(mask[..., None], tensor)
            )
        else:
            identity = self.identity(
                self.shape,
                rotation_type=type(self.rot),
                device=self.device,
                dtype=self.dtype,
            ).tensor
            return Affine3D.from_tensor(identity.where(mask[..., None], self.tensor))

    def apply(self, p: torch.Tensor) -> torch.Tensor:
        return self.rot.apply(p) + self.trans

    def invert(self):
        inv_rot = self.rot.invert()
        return Affine3D(trans=-inv_rot.apply(self.trans), rot=inv_rot)

    @property
    def tensor(self) -> torch.Tensor:
        return torch.cat([self.rot.tensor, self.trans], dim=-1)

    @staticmethod
    def from_tensor(t: torch.Tensor) -> "Affine3D":
        match t.shape[-1]:
            case 4:
                # Assume tensor 4x4 for backward compat with alphafold
                trans = t[..., :3, 3]
                rot = RotationMatrix(t[..., :3, :3])
            case 6:
                # Assume quaternion representation with real part = 1
                trans = t[..., -3:]
                rot = RotationQuat(F.pad(t[..., :3], (1, 0), value=1))
            case 7:
                trans = t[..., -3:]
                rot = RotationQuat(t[..., :4])
            case 12:
                trans = t[..., -3:]
                rot = RotationMatrix(t[..., :-3].unflatten(-1, (3, 3)))
            case _:
                raise RuntimeError(
                    f"Cannot detect rotation fromat from {t.shape[-1] - 3}-d flat vector"
                )
        return Affine3D(trans, rot)

    @staticmethod
    def from_tensor_pair(t: torch.Tensor, r: torch.Tensor) -> "Affine3D":
        return Affine3D(t, RotationMatrix(r))

    @staticmethod
    def from_graham_schmidt(
        neg_x_axis: torch.Tensor,
        origin: torch.Tensor,
        xy_plane: torch.Tensor,
        eps: float = 1e-10,
    ):
        # The arguments of this function is for parity with AlphaFold
        x_axis = origin - neg_x_axis
        xy_plane = xy_plane - origin
        return Affine3D(
            trans=origin, rot=RotationMatrix.from_graham_schmidt(x_axis, xy_plane, eps)
        )

    @staticmethod
    def cat(affines: list["Affine3D"], dim: int = 0):
        if dim < 0:
            dim = len(affines[0].shape) + dim
        return Affine3D.from_tensor(torch.cat([x.tensor for x in affines], dim=dim))


def _quat_mult(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def _quat_rotation(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """
    Rotates p by quaternion q. Usual torch rules for broadcasting apply.

    Args:
        q: Quaternions as tensor of shape (..., 4), real part first.
        p: Points as tensor of shape (..., 3)

    Returns:
        The rotated version of p, of shape (..., 3)
    """
    aw, ax, ay, az = torch.unbind(q, -1)
    bx, by, bz = torch.unbind(p, -1)
    # fmt: off
    ow =         - ax * bx - ay * by - az * bz
    ox = aw * bx           + ay * bz - az * by
    oy = aw * by - ax * bz           + az * bx
    oz = aw * bz + ax * by - ay * bx
    # fmt: on
    q_mul_pts = torch.stack((ow, ox, oy, oz), -1)
    return _quat_mult(q_mul_pts, _quat_invert(q))[..., 1:]


def _quat_invert(q: torch.Tensor):
    return q * torch.tensor([1, -1, -1, -1], device=q.device)


def _sqrt_subgradient(x: torch.Tensor) -> torch.Tensor:
    # Returns torch.sqrt(torch.max(0, x)) but with a zero subgradient where x is 0.
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def _graham_schmidt(x_axis: torch.Tensor, xy_plane: torch.Tensor, eps: float = 1e-12):
    # A low eps here is necessary for good stability!
    with fp32_autocast_context(x_axis.device.type):
        e1 = xy_plane

        denom = torch.sqrt((x_axis**2).sum(dim=-1, keepdim=True) + eps)
        x_axis = x_axis / denom
        dot = (x_axis * e1).sum(dim=-1, keepdim=True)
        e1 = e1 - x_axis * dot
        denom = torch.sqrt((e1**2).sum(dim=-1, keepdim=True) + eps)
        e1 = e1 / denom
        e2 = torch.cross(x_axis, e1, dim=-1)

        rots = torch.stack([x_axis, e1, e2], dim=-1)

        return rots


def build_affine3d_from_coordinates(
    coords: torch.Tensor,  # (N, CA, C).
) -> tuple[Affine3D, torch.Tensor]:
    _MAX_SUPPORTED_DISTANCE = 1e6
    coord_mask = torch.all(
        torch.all(torch.isfinite(coords) & (coords < _MAX_SUPPORTED_DISTANCE), dim=-1),
        dim=-1,
    )

    def atom3_to_backbone_affine(bb_positions: torch.Tensor) -> Affine3D:
        N, CA, C = bb_positions.unbind(dim=-2)
        return Affine3D.from_graham_schmidt(C, CA, N)

    coords = coords.clone().float()
    coords[~coord_mask] = 0

    # NOTE(thayes): If you have already normalized the coordinates, then
    # the black hole affine translations will be zeros and the rotations will be
    # the identity.
    average_per_n_ca_c = coords.masked_fill(~coord_mask[..., None, None], 0).sum(1) / (
        coord_mask.sum(-1)[..., None, None] + 1e-8
    )
    affine_from_average = atom3_to_backbone_affine(
        average_per_n_ca_c.float()
    ).as_matrix()

    B, S, _, _ = coords.shape
    assert isinstance(B, int)
    assert isinstance(S, int)
    affine_rot_mats = affine_from_average.rot.tensor[..., None, :].expand(B, S, 9)
    affine_trans = affine_from_average.trans[..., None, :].expand(B, S, 3)

    # We use the identity rotation whereever we have no coordinates. This is
    # important because otherwise the rotation matrices will be all zeros, which
    # will cause collapse in the distance/direction attention mechanism.
    identity_rot = RotationMatrix.identity(
        (B, S), dtype=torch.float32, device=coords.device, requires_grad=False
    )
    affine_rot_mats = affine_rot_mats.where(
        coord_mask.any(-1)[..., None, None], identity_rot.tensor
    )
    black_hole_affine = Affine3D(affine_trans, RotationMatrix(affine_rot_mats))

    affine = atom3_to_backbone_affine(coords.float())
    affine = Affine3D.from_tensor(
        affine.tensor.where(coord_mask[..., None], black_hole_affine.tensor)
    )

    return affine, coord_mask
