// The Anvil — where Ingots are shaped into an agent design.
//
// Place Ingots on the Anvil, arrange them, and link them together
// to define data flow and dependencies.

use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};

/// A slot on the Anvil — a placed Ingot being worked on.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnvilSlot {
    /// Reference to the Ingot this slot is based on.
    pub ingot_id: String,
    /// Position on the anvil (for UI layout).
    pub position: (u16, u16),
    /// Custom properties applied to this instance.
    pub properties: HashMap<String, serde_json::Value>,
}

/// A link between two slots on the Anvil.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Link {
    pub from: String,
    pub to: String,
    /// Label describing the connection.
    pub label: String,
}

/// The Anvil — the active workspace.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Anvil {
    /// Placed slots, keyed by a unique slot ID.
    pub slots: HashMap<String, AnvilSlot>,
    /// Connections between slots.
    pub links: Vec<Link>,
}

impl Anvil {
    pub fn new() -> Self {
        Self::default()
    }

    /// Place an Ingot on the Anvil at a position.
    pub fn place(&mut self, slot_id: &str, ingot_id: &str, position: (u16, u16)) {
        self.slots.insert(
            slot_id.to_string(),
            AnvilSlot {
                ingot_id: ingot_id.to_string(),
                position,
                properties: HashMap::new(),
            },
        );
    }

    /// Remove a slot from the Anvil.
    pub fn remove(&mut self, slot_id: &str) {
        self.slots.remove(slot_id);
        self.links.retain(|l| l.from != slot_id && l.to != slot_id);
    }

    /// Add a link between two slots.
    pub fn link(&mut self, from: &str, to: &str, label: &str) {
        if self.slots.contains_key(from) && self.slots.contains_key(to) {
            self.links.push(Link {
                from: from.to_string(),
                to: to.to_string(),
                label: label.to_string(),
            });
        }
    }

    /// Remove a link.
    pub fn unlink(&mut self, from: &str, to: &str) {
        self.links.retain(|l| !(l.from == from && l.to == to));
    }

    /// Set a custom property on a slot.
    pub fn set_property(&mut self, slot_id: &str, key: &str, value: serde_json::Value) {
        if let Some(slot) = self.slots.get_mut(slot_id) {
            slot.properties.insert(key.to_string(), value);
        }
    }

    /// List all slot IDs.
    pub fn slot_ids(&self) -> Vec<&str> {
        self.slots.keys().map(|s| s.as_str()).collect()
    }

    /// Check for circular links.
    pub fn has_cycles(&self) -> bool {
        let mut visited = HashSet::new();
        let mut stack = Vec::new();

        for slot_id in self.slots.keys() {
            if visited.contains(slot_id) {
                continue;
            }
            stack.clear();
            Self::dfs(self, slot_id, &mut visited, &mut stack);
        }

        !stack.is_empty()
    }

    fn dfs(anvil: &Anvil, node: &str, visited: &mut HashSet<String>, stack: &mut Vec<String>) -> bool {
        let node_owned = node.to_string();
        if stack.contains(&node_owned) {
            stack.push(node_owned.clone());
            return true;
        }
        if visited.contains(&node_owned) {
            return false;
        }

        visited.insert(node_owned.clone());
        stack.push(node_owned.clone());

        for link in &anvil.links {
            if link.from == node {
                if Self::dfs(anvil, &link.to, visited, stack) {
                    return true;
                }
            }
        }

        stack.pop();
        false
    }

    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    pub fn link_count(&self) -> usize {
        self.links.len()
    }
}
