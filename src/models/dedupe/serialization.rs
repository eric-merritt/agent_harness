use crate::models::dedupe::types::Sandbag;
use std::convert::TryInto;

/// Binary file layout (little-endian, no padding):
///
/// ┌──────────────────────────────────────────────────────────────┐
/// │  Header (fixed 8 bytes)                                      │
///  │  id: 4B ("SBAG" — identifies this as a sandbag blob)         │
///  │  version: 2B (major:minor, currently 0x00_03)                │
///  │  reserved: 2B (padding for future extensions)                │
/// └──────────────────────────────────────────────────────────────┘
/// ┌──────────────────────────────────────────────────────────────┐
/// │  Payload sections (length-prefixed, ordered)                  │
///  │                                                               │
///  │  [A] count: u32                                              │
///  │  [B] scale: f32                                              │
///  │  [C] prefix_digits: u32                                      │
///  │  [D] outliers: u32(count) × [u32(pos) + f32(val)]            │
///  │  [E] unique_prefixes: u32(count) × [u8]                      │
///  │  [F] unique_tails: u32(count) × [u32]                        │
///  │  [G] manifest: u32(count) × [u16(p_idx) + u16(t_idx)]       │
///  │  [H] signs: u32(bytes) × [u8] (bitvector, 1 bit per element) │
/// └──────────────────────────────────────────────────────────────┘
///
/// Sections are NOT offset-indexed — you must scan sequentially.

const ID: [u8; 4] = *b"SBAG";
const VERSION_MAJOR: u8 = 0;
const VERSION_MINOR: u8 = 3;

impl Sandbag {
    pub fn estimated_size(&self) -> usize {
        8 // header
        + 4 // count
        + 4 // scale
        + 4 // prefix_digits
        + 4 // outlier_count
        + self.outliers.len() * 8
        + 4 // unique_prefixes count
        + self.unique_prefixes.len() // u8 each
        + 4 // unique_tails count
        + self.unique_tails.len() * 4 // u32 each
        + 4 // manifest count
        + self.manifest.len() * 4 // u16 + u16
        + 4 // signs byte count
        + self.signs.len()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.estimated_size());

        // Header
        bytes.extend_from_slice(&ID);
        bytes.push(VERSION_MAJOR);
        bytes.push(VERSION_MINOR);
        bytes.extend_from_slice(&[0u8; 2]);

        // [A] count
        bytes.extend_from_slice(&(self.count as u32).to_le_bytes());

        // [B] scale
        bytes.extend_from_slice(&self.scale.to_le_bytes());

        // [C] prefix_digits
        bytes.extend_from_slice(&(self.prefix_digits as u32).to_le_bytes());

        // [D] outliers
        bytes.extend_from_slice(&(self.outliers.len() as u32).to_le_bytes());
        for &(idx, val) in &self.outliers {
            bytes.extend_from_slice(&(idx as u32).to_le_bytes());
            bytes.extend_from_slice(&val.to_le_bytes());
        }

        // [E] unique_prefixes (u8 each)
        bytes.extend_from_slice(&(self.unique_prefixes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.unique_prefixes);

        // [F] unique_tails (u32 each)
        bytes.extend_from_slice(&(self.unique_tails.len() as u32).to_le_bytes());
        for &v in &self.unique_tails {
            bytes.extend_from_slice(&v.to_le_bytes());
        }

        // [G] manifest (u16 p_idx + u16 t_idx per entry)
        bytes.extend_from_slice(&(self.manifest.len() as u32).to_le_bytes());
        for &(p_idx, t_idx) in &self.manifest {
            bytes.extend_from_slice(&p_idx.to_le_bytes());
            bytes.extend_from_slice(&t_idx.to_le_bytes());
        }

        // [H] signs bitvector
        bytes.extend_from_slice(&(self.signs.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.signs);

        bytes
    }

    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 8 {
            return None;
        }
        if data[0..4] != ID {
            return None;
        }
        let major = data[4];
        if major != VERSION_MAJOR {
            return None;
        }

        let mut pos = 8;

        let mut read = |buf: &mut [u8], n: usize, data: &[u8], pos: &mut usize| -> Option<()> {
            let end = pos.saturating_add(n);
            if end > data.len() { return None; }
            buf.copy_from_slice(&data[*pos..end]);
            *pos = end;
            Some(())
        };

        // [A] count
        let mut buf4 = [0u8; 4];
        read(&mut buf4, 4, data, &mut pos)?;
        let count = u32::from_le_bytes(buf4) as usize;

        // [B] scale
        read(&mut buf4, 4, data, &mut pos)?;
        let scale = f32::from_le_bytes(buf4);

        // [C] prefix_digits
        read(&mut buf4, 4, data, &mut pos)?;
        let prefix_digits = u32::from_le_bytes(buf4) as usize;

        // [D] outliers
        read(&mut buf4, 4, data, &mut pos)?;
        let outlier_count = u32::from_le_bytes(buf4) as usize;
        let outliers_byte_len = outlier_count * 8;
        let outliers_data = data.get(pos..pos + outliers_byte_len)?;
        pos += outliers_byte_len;

        let mut outliers = Vec::with_capacity(outlier_count);
        for chunk in outliers_data.chunks_exact(8) {
            let idx = u32::from_le_bytes(chunk[0..4].try_into().ok()?) as usize;
            let val = f32::from_le_bytes(chunk[4..8].try_into().ok()?);
            outliers.push((idx, val));
        }

        // [E] unique_prefixes (u8 each)
        read(&mut buf4, 4, data, &mut pos)?;
        let up_count = u32::from_le_bytes(buf4) as usize;
        let up_data = data.get(pos..pos + up_count)?;
        pos += up_count;
        let unique_prefixes: Vec<u8> = up_data.to_vec();

        // [F] unique_tails (u32 each)
        read(&mut buf4, 4, data, &mut pos)?;
        let ut_count = u32::from_le_bytes(buf4) as usize;
        let ut_byte_len = ut_count * 4;
        let ut_data = data.get(pos..pos + ut_byte_len)?;
        pos += ut_byte_len;

        let mut unique_tails = Vec::with_capacity(ut_count);
        for chunk in ut_data.chunks_exact(4) {
            unique_tails.push(u32::from_le_bytes(chunk.try_into().ok()?));
        }

        // [G] manifest (u16 p_idx + u16 t_idx per entry)
        read(&mut buf4, 4, data, &mut pos)?;
        let manifest_count = u32::from_le_bytes(buf4) as usize;
        let manifest_byte_len = manifest_count * 4;
        let manifest_data = data.get(pos..pos + manifest_byte_len)?;
        pos += manifest_byte_len;

        let mut manifest = Vec::with_capacity(manifest_count);
        for chunk in manifest_data.chunks_exact(4) {
            let p_idx = u16::from_le_bytes(chunk[0..2].try_into().ok()?);
            let t_idx = u16::from_le_bytes(chunk[2..4].try_into().ok()?);
            manifest.push((p_idx, t_idx));
        }

        // [H] signs bitvector
        read(&mut buf4, 4, data, &mut pos)?;
        let signs_len = u32::from_le_bytes(buf4) as usize;
        let signs_data = data.get(pos..pos + signs_len)?;
        pos += signs_len;
        let signs: Vec<u8> = signs_data.to_vec();

        Some(Self {
            scale,
            outliers,
            count,
            prefix_digits,
            unique_prefixes,
            unique_tails,
            manifest,
            signs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sandbag_roundtrip() {
        let sb = Sandbag {
            scale: 0.00012345f32,
            outliers: vec![(100, 1.5f32), (200, -0.7f32)],
            count: 7,
            prefix_digits: 2,
            unique_prefixes: vec![27u8, 27u8],
            unique_tails: vec![9453000u32, 100000u32],
            manifest: vec![
                (0u16, 0u16),
                (1u16, 1u16),
                (0u16, 0u16),
            ],
            signs: vec![0b00000101u8, 0u8], // bits 0,2 negative
        };

        let bytes = sb.to_bytes();
        eprintln!("serialized size: {} bytes", bytes.len());
        eprintln!("header: {:02X?}", &bytes[0..8]);

        let sb2 = Sandbag::from_bytes(&bytes).expect("deserialization failed");

        assert_eq!(sb2.scale, sb.scale, "scale mismatch");
        assert_eq!(sb2.count, sb.count, "count mismatch");
        assert_eq!(sb2.prefix_digits, sb.prefix_digits, "prefix_digits mismatch");
        assert_eq!(sb2.outliers, sb.outliers, "outliers mismatch");
        assert_eq!(sb2.unique_prefixes, sb.unique_prefixes, "unique_prefixes mismatch");
        assert_eq!(sb2.unique_tails, sb.unique_tails, "unique_tails mismatch");
        assert_eq!(sb2.manifest, sb.manifest, "manifest mismatch");
        assert_eq!(sb2.signs, sb.signs, "signs mismatch");
    }

    #[test]
    fn test_sandbag_empty() {
        let sb = Sandbag {
            scale: 1.0,
            outliers: vec![],
            count: 0,
            prefix_digits: 2,
            unique_prefixes: vec![],
            unique_tails: vec![],
            manifest: vec![],
            signs: vec![],
        };
        let bytes = sb.to_bytes();
        let sb2 = Sandbag::from_bytes(&bytes).expect("empty roundtrip failed");
        assert_eq!(sb2.count, 0);
        assert_eq!(sb2.unique_prefixes.len(), 0);
        assert_eq!(sb2.manifest.len(), 0);
    }

    #[test]
    fn test_bad_magic_rejected() {
        let mut bytes = Sandbag {
            scale: 1.0,
            outliers: vec![],
            count: 0,
            prefix_digits: 2,
            unique_prefixes: vec![],
            unique_tails: vec![],
            manifest: vec![],
            signs: vec![],
        }
        .to_bytes();
        bytes[0] = 0;
        assert!(Sandbag::from_bytes(&bytes).is_none(), "should reject bad magic");
    }

    #[test]
    fn test_major_version_mismatch() {
        let mut bytes = Sandbag {
            scale: 1.0,
            outliers: vec![],
            count: 0,
            prefix_digits: 2,
            unique_prefixes: vec![],
            unique_tails: vec![],
            manifest: vec![],
            signs: vec![],
        }
        .to_bytes();
        bytes[4] = 1;
        assert!(Sandbag::from_bytes(&bytes).is_none(), "should reject major version bump");
    }
}
