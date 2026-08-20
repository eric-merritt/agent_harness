use crate::models::dedup::types::UniqueTail;


#[derive(Clone, Debug)]
pub struct DedupCountTensor {
    pub prefixes: Vec<u16>,      
    pub prefix_counts: Vec<u32>,
    pub unique_tails: Vec<UniqueTail>,
    pub count: usize,
    pub prefix_digits: usize,
    pub tail_digits: usize,
    pub avg_precision_lost: f32,
}

impl DedupCountTensor {
    pub const TOTAL_DIGITS: usize = 7;

    pub fn unique_tail_count(&self) -> usize {
        self.unique_tails.len()
    }

    pub fn shared_tail_weights(&self) -> usize {
        self.unique_tails.iter()
            .filter(|ut| ut.repeat_count > 1)
            .map(|ut| ut.repeat_count as usize)
            .sum()
    }

    pub fn compressed_bytes(&self) -> usize {
        let header = 4 + 4 + 4 + 4 + 4; 
        let front = self.prefixes.len() * 2 + self.unique_tails.len() * 2;     
        let flags = 3; 
        header + front + flags
    }
}