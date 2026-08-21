/// Max clusters that fit in 7 bits (high bit is the inverse flag).
pub const MAX_CLUSTERS: usize = 127;
/// Max tails per cluster that fit in 1 byte (0xFF is the sentinel).
pub const MAX_TAILS_PER_CLUSTER: usize = 254;