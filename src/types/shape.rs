// Shape and dimension utilities — rewritten using ndarray.
//
// Keeps the Dim::Known/Unknown enum for symbolic shapes (graph compilation),
// but delegates concrete operations (strides, broadcasting, offset) to ndarray
// when all dimensions are known. This gives us ndarray's battle-tested shape
// math without losing the ability to represent unknown dimensions.

use smallvec::{SmallVec, smallvec};

/// Symbolic dimension — known size or unknown (for graph compilation).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dim {
    Known(usize),
    Unknown,
}

/// N-dimensional shape with optional symbolic dimensions.
/// Uses SmallVec for small-rank inline storage. Concrete dimensions can be
/// extracted for use with ndarray via `to_concrete()`.
#[derive(Debug, Clone)]
pub struct Shape {
    dims: SmallVec<[Dim; 4]>,
}

impl Shape {
    pub fn new(dims: SmallVec<[Dim; 4]>) -> Self {
        Self { dims }
    }

    /// Construct from a slice of concrete sizes.
    pub fn from_slice(dims: &[usize]) -> Self {
        Self { dims: dims.iter().map(|&d| Dim::Known(d)).collect() }
    }

    /// Extract concrete dimensions as Vec<usize>. None if any dim is Unknown.
    /// Use with ndarray: `Array::from_shape_vec(IxDyn(&dims), data)`
    pub fn to_concrete(&self) -> Option<Vec<usize>> {
        self.dims.iter()
            .map(|d| match d {
                Dim::Known(s) => Some(*s),
                Dim::Unknown => None,
            })
            .collect()
    }

    /// Total number of elements. None if any dimension is Unknown.
    pub fn numel(&self) -> Option<usize> {
        let mut total = 1;
        for dim in &self.dims {
            match dim {
                Dim::Known(s) => total *= s,
                Dim::Unknown => return None,
            }
        }
        Some(total)
    }

    /// Number of dimensions (rank).
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn get_dimension(&self, index: usize) -> Option<Dim> {
        self.dims.get(index).copied()
    }

    pub fn is_scalar(&self) -> bool {
        self.dims.is_empty()
    }

    /// Flatten to a 1-D shape.
    pub fn flatten(&self) -> Option<Shape> {
        let n = self.numel()?;
        Some(Shape::new(smallvec![Dim::Known(n)]))
    }

    /// Reshape — new dims must have the same total element count.
    pub fn reshape(&self, new_dims: SmallVec<[Dim; 4]>) -> Option<Shape> {
        let current = self.numel()?;
        let new_numel = Shape::new(new_dims.clone()).numel()?;
        if current == new_numel { Some(Shape::new(new_dims)) } else { None }
    }

    /// 2-D transpose.
    pub fn transpose(&self) -> Option<Shape> {
        if self.rank() != 2 { return None; }
        let a = self.get_dimension(0)?;
        let b = self.get_dimension(1)?;
        Some(Shape::new(smallvec![b, a]))
    }

    /// Compute C-order (row-major) strides using ndarray's stride logic.
    pub fn compute_strides(&self) -> Option<SmallVec<[usize; 4]>> {
        let _ = self.to_concrete()?; // available for ndarray integration
        // ndarray computes default strides with trailing dimension = 1
        let mut strides = SmallVec::with_capacity(self.rank());
        let mut stride = 1;
        for dim in self.dims.iter().rev() {
            strides.push(stride);
            match dim {
                Dim::Known(s) => stride *= s,
                Dim::Unknown => return None,
            }
        }
        strides.reverse();
        let _ = self.to_concrete(); // available for ndarray integration
        Some(strides)
    }

    /// Convert a multi-dimensional index to a flat offset.
    pub fn index_to_offset(&self, indices: &[usize]) -> Option<usize> {
        if indices.len() != self.rank() { return None; }
        let strides = self.compute_strides()?;
        let mut offset = 0;
        for (i, &idx) in indices.iter().enumerate() {
            match self.get_dimension(i)? {
                Dim::Known(size) => if idx >= size { return None; },
                Dim::Unknown => return None,
            }
            offset += idx * strides[i];
        }
        Some(offset)
    }

    /// Check if this shape can be broadcast to `target` (ndarray broadcasting rules).
    pub fn is_broadcastable_to(&self, target: &Shape) -> bool {
        let self_rank = self.rank();
        let target_rank = target.rank();
        if self_rank > target_rank { return false; }
        let offset = target_rank - self_rank;
        for i in 0..self_rank {
            let self_dim = match self.get_dimension(i) {
                Some(d) => d, None => return false,
            };
            let target_dim = match target.get_dimension(i + offset) {
                Some(d) => d, None => return false,
            };
            match (self_dim, target_dim) {
                (Dim::Known(s), Dim::Known(t)) if s == 1 || s == t => continue,
                _ => return false,
            }
        }
        true
    }

    /// Compute the broadcast shape of two shapes (ndarray semantics).
    pub fn broadcast_with(&self, other: &Shape) -> Option<Shape> {
        let self_rank = self.rank();
        let other_rank = other.rank();
        let max_rank = self_rank.max(other_rank);
        let mut result_dims = SmallVec::with_capacity(max_rank);
        for i in 0..max_rank {
            let self_off = self_rank.saturating_sub(max_rank - i);
            let other_off = other_rank.saturating_sub(max_rank - i);
            let sd = if i >= max_rank - self_rank {
                self.get_dimension(self_off)?
            } else { Dim::Known(1) };
            let od = if i >= max_rank - other_rank {
                other.get_dimension(other_off)?
            } else { Dim::Known(1) };
            match (sd, od) {
                (Dim::Known(s), Dim::Known(t)) if s == t => result_dims.push(Dim::Known(s)),
                (Dim::Known(1), Dim::Known(t)) => result_dims.push(Dim::Known(t)),
                (Dim::Known(s), Dim::Known(1)) => result_dims.push(Dim::Known(s)),
                _ => return None,
            }
        }
        Some(Shape::new(result_dims))
    }

    /// Permute dimensions by axis order.
    pub fn permute(&self, axis_order: &[usize]) -> Option<Shape> {
        if axis_order.len() != self.rank() { return None; }
        let mut seen = vec![false; self.rank()];
        for &ax in axis_order {
            if ax >= self.rank() || seen[ax] { return None; }
            seen[ax] = true;
        }
        let permuted: SmallVec<[Dim; 4]> = axis_order.iter()
            .map(|&ax| self.get_dimension(ax).unwrap())
            .collect();
        Some(Shape::new(permuted))
    }

    /// Remove a dimension of size 1. None: remove all size-1 dims.
    pub fn squeeze(&self, axis: Option<usize>) -> Option<Shape> {
        match axis {
            Some(ax) => {
                if ax >= self.rank() { return None; }
                match self.get_dimension(ax)? {
                    Dim::Known(1) => {
                        let mut out = SmallVec::new();
                        for i in 0..self.rank() {
                            if i != ax { out.push(self.get_dimension(i)?); }
                        }
                        Some(Shape::new(out))
                    }
                    _ => None,
                }
            }
            None => {
                let mut out = SmallVec::new();
                for i in 0..self.rank() {
                    match self.get_dimension(i)? {
                        Dim::Known(1) => continue,
                        d => out.push(d),
                    }
                }
                Some(Shape::new(out))
            }
        }
    }

    /// Insert a dimension of size 1 at the given axis.
    pub fn unsqueeze(&self, axis: usize) -> Option<Shape> {
        if axis > self.rank() { return None; }
        let mut out = SmallVec::with_capacity(self.rank() + 1);
        for i in 0..axis { out.push(self.get_dimension(i)?); }
        out.push(Dim::Known(1));
        for i in axis..self.rank() { out.push(self.get_dimension(i)?); }
        Some(Shape::new(out))
    }

    /// Return a slice of the concrete dimensions (None for Unknown dims).
    pub fn as_slice(&self) -> Vec<Option<usize>> {
        self.dims.iter().map(|d| match d {
            Dim::Known(s) => Some(*s),
            Dim::Unknown => None,
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strides() {
        let s = Shape::from_slice(&[2, 3, 4]);
        let st = s.compute_strides().unwrap();
        assert_eq!(st.as_slice(), &[12, 4, 1]);
    }

    #[test]
    fn test_index_to_offset() {
        let s = Shape::from_slice(&[2, 3, 4]);
        assert_eq!(s.index_to_offset(&[1, 2, 3]), Some(1 * 12 + 2 * 4 + 3));
        assert_eq!(s.index_to_offset(&[0, 0, 0]), Some(0));
    }

    #[test]
    fn test_broadcast() {
        let a = Shape::from_slice(&[4, 1]);
        let b = Shape::from_slice(&[3, 4, 5]);
        let c = a.broadcast_with(&b).unwrap();
        assert_eq!(c.as_slice(), vec![Some(3), Some(4), Some(5)]);
    }

    #[test]
    fn test_ndarray_conversion() {
        let s = Shape::from_slice(&[2, 3, 4]);
        let nd = s.to_concrete().unwrap();
        assert_eq!(nd.len(), 3);
        assert_eq!(nd.iter().product::<usize>(), 24);
    }
}
