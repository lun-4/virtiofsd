// Copyright 2026 The Virtiofs Project Developers
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

//! Registry for managing multiple shares with path specificity.
//!
//! The [`ShareRegistry`] manages a collection of [`Share`]s and provides
//! path-based permission lookup with specificity rules: when multiple shares
//! overlap, the most specific (deepest) share wins.

use crate::share::{Share, ShareMode};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

/// Unique identifier for a share.
pub type ShareId = u64;

/// A share entry with its unique identifier.
#[derive(Debug, Clone)]
pub struct ShareEntry {
    /// Unique identifier for this share.
    pub id: ShareId,
    /// The share definition.
    pub share: Share,
}

/// Registry managing multiple shares with thread-safe access.
///
/// Provides path-based permission lookups that respect specificity rules:
/// more specific paths (deeper in the hierarchy) take precedence over
/// less specific ones.
pub struct ShareRegistry {
    /// All registered shares, keyed by their canonical path.
    shares: RwLock<HashMap<PathBuf, ShareEntry>>,
    /// Counter for generating unique share IDs.
    next_id: AtomicU64,
}

impl ShareRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        ShareRegistry {
            shares: RwLock::new(HashMap::new()),
            next_id: AtomicU64::new(1),
        }
    }

    /// Create a registry with initial shares.
    pub fn with_shares(shares: impl IntoIterator<Item = Share>) -> Self {
        let registry = Self::new();
        for share in shares {
            registry.add_share(share);
        }
        registry
    }

    /// Add a share to the registry.
    ///
    /// If a share with the same path already exists, it will be replaced.
    /// Returns the ID of the added share.
    pub fn add_share(&self, share: Share) -> ShareId {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let path = share.path().to_path_buf();
        let entry = ShareEntry { id, share };

        let mut shares = self.shares.write().unwrap();
        shares.insert(path, entry);
        id
    }

    /// Remove a share by its path.
    ///
    /// Returns the removed share entry if it existed.
    pub fn remove_share(&self, path: &Path) -> Option<ShareEntry> {
        let mut shares = self.shares.write().unwrap();
        shares.remove(path)
    }

    /// Remove a share by its ID.
    ///
    /// Returns the removed share entry if it existed.
    pub fn remove_share_by_id(&self, id: ShareId) -> Option<ShareEntry> {
        let mut shares = self.shares.write().unwrap();
        let path = shares
            .iter()
            .find(|(_, e)| e.id == id)
            .map(|(p, _)| p.clone())?;
        shares.remove(&path)
    }

    /// Get the share for a specific path (exact match).
    pub fn get_share(&self, path: &Path) -> Option<ShareEntry> {
        let shares = self.shares.read().unwrap();
        shares.get(path).cloned()
    }

    /// List all shares.
    pub fn list_shares(&self) -> Vec<ShareEntry> {
        let shares = self.shares.read().unwrap();
        shares.values().cloned().collect()
    }

    /// Returns the number of registered shares.
    pub fn len(&self) -> usize {
        self.shares.read().unwrap().len()
    }

    /// Returns true if no shares are registered.
    pub fn is_empty(&self) -> bool {
        self.shares.read().unwrap().is_empty()
    }

    /// Find the most specific share that contains the given path.
    ///
    /// When multiple shares contain a path, the one with the deepest
    /// (most specific) path takes precedence.
    ///
    /// Returns `None` if the path is not under any share.
    pub fn find_share(&self, path: &Path) -> Option<ShareEntry> {
        let shares = self.shares.read().unwrap();

        shares
            .values()
            .filter(|entry| entry.share.contains(path))
            .max_by_key(|entry| entry.share.depth())
            .cloned()
    }

    /// Get the permission for a path based on the most specific share.
    ///
    /// Returns `None` if the path is not under any share.
    pub fn get_permission(&self, path: &Path) -> Option<ShareMode> {
        self.find_share(path).map(|entry| entry.share.mode())
    }

    /// Check if a path is visible (under any share).
    pub fn is_visible(&self, path: &Path) -> bool {
        let shares = self.shares.read().unwrap();
        shares.values().any(|entry| entry.share.contains(path))
    }

    /// Check if a path is an ancestor of any share.
    ///
    /// This is used during directory listing to show directories
    /// that lead to a share, even if they're not directly shared.
    pub fn is_ancestor_of_share(&self, path: &Path) -> bool {
        let shares = self.shares.read().unwrap();
        shares
            .values()
            .any(|entry| entry.share.path().starts_with(path) && entry.share.path() != path)
    }

    /// Check if a path should be visible in directory listings.
    ///
    /// A path is visible if:
    /// 1. It is under a share, OR
    /// 2. It is an ancestor of a share (to allow navigation to shares)
    pub fn is_path_visible(&self, path: &Path) -> bool {
        self.is_visible(path) || self.is_ancestor_of_share(path)
    }

    /// Get the share entries that are direct children of the given path.
    ///
    /// Used for directory listing to find shares that should appear
    /// as entries in this directory.
    pub fn get_child_shares(&self, parent: &Path) -> Vec<ShareEntry> {
        let shares = self.shares.read().unwrap();
        shares
            .values()
            .filter(|entry| {
                entry
                    .share
                    .path()
                    .parent()
                    .map(|p| p == parent)
                    .unwrap_or(false)
            })
            .cloned()
            .collect()
    }

    /// Filter a list of directory entry names to only those that are visible.
    ///
    /// Given a parent path and a list of entry names, returns only the names
    /// that should be visible based on the share configuration.
    pub fn filter_visible_entries(&self, parent: &Path, entries: &[&str]) -> Vec<String> {
        entries
            .iter()
            .filter(|name| {
                let full_path = parent.join(name);
                self.is_path_visible(&full_path)
            })
            .map(|s| s.to_string())
            .collect()
    }
}

impl Default for ShareRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_share(path: &str, mode: ShareMode) -> Share {
        Share::new_unchecked(PathBuf::from(path), mode)
    }

    #[test]
    fn test_add_and_get_share() {
        let registry = ShareRegistry::new();
        let share = make_share("/home/user", ShareMode::ReadOnly);
        let id = registry.add_share(share.clone());

        let entry = registry.get_share(Path::new("/home/user")).unwrap();
        assert_eq!(entry.id, id);
        assert_eq!(entry.share.path(), Path::new("/home/user"));
        assert_eq!(entry.share.mode(), ShareMode::ReadOnly);
    }

    #[test]
    fn test_remove_share() {
        let registry = ShareRegistry::new();
        registry.add_share(make_share("/home/user", ShareMode::ReadOnly));

        assert!(registry.get_share(Path::new("/home/user")).is_some());

        let removed = registry.remove_share(Path::new("/home/user"));
        assert!(removed.is_some());
        assert!(registry.get_share(Path::new("/home/user")).is_none());
    }

    #[test]
    fn test_find_share_specificity() {
        let registry = ShareRegistry::new();
        registry.add_share(make_share("/home/user", ShareMode::ReadOnly));
        registry.add_share(make_share("/home/user/project", ShareMode::ReadWrite));

        // Path under the more specific share
        let entry = registry
            .find_share(Path::new("/home/user/project/file.txt"))
            .unwrap();
        assert_eq!(entry.share.path(), Path::new("/home/user/project"));
        assert_eq!(entry.share.mode(), ShareMode::ReadWrite);

        // Path under the less specific share
        let entry = registry
            .find_share(Path::new("/home/user/documents/file.txt"))
            .unwrap();
        assert_eq!(entry.share.path(), Path::new("/home/user"));
        assert_eq!(entry.share.mode(), ShareMode::ReadOnly);
    }

    #[test]
    fn test_permission_resolution() {
        let registry = ShareRegistry::new();
        registry.add_share(make_share("/home/user", ShareMode::ReadOnly));
        registry.add_share(make_share("/home/user/project", ShareMode::ReadWrite));

        // Permissions follow specificity
        assert_eq!(
            registry.get_permission(Path::new("/home/user/project/src")),
            Some(ShareMode::ReadWrite)
        );
        assert_eq!(
            registry.get_permission(Path::new("/home/user/documents")),
            Some(ShareMode::ReadOnly)
        );
        assert_eq!(
            registry.get_permission(Path::new("/home/user")),
            Some(ShareMode::ReadOnly)
        );

        // Paths outside shares have no permission
        assert_eq!(registry.get_permission(Path::new("/home/other")), None);
        assert_eq!(registry.get_permission(Path::new("/etc")), None);
    }

    #[test]
    fn test_visibility() {
        let registry = ShareRegistry::new();
        registry.add_share(make_share("/home/user", ShareMode::ReadOnly));
        registry.add_share(make_share("/opt/data", ShareMode::ReadWrite));

        // Paths under shares are visible
        assert!(registry.is_visible(Path::new("/home/user")));
        assert!(registry.is_visible(Path::new("/home/user/file.txt")));
        assert!(registry.is_visible(Path::new("/opt/data")));

        // Paths outside shares are not visible
        assert!(!registry.is_visible(Path::new("/home/other")));
        assert!(!registry.is_visible(Path::new("/etc")));
    }

    #[test]
    fn test_ancestor_visibility() {
        let registry = ShareRegistry::new();
        registry.add_share(make_share("/home/user/project", ShareMode::ReadWrite));

        // Ancestors of shares should be visible for navigation
        assert!(registry.is_ancestor_of_share(Path::new("/home")));
        assert!(registry.is_ancestor_of_share(Path::new("/home/user")));

        // The share itself is not its own ancestor
        assert!(!registry.is_ancestor_of_share(Path::new("/home/user/project")));

        // Children of shares are not ancestors
        assert!(!registry.is_ancestor_of_share(Path::new("/home/user/project/src")));

        // Unrelated paths are not ancestors
        assert!(!registry.is_ancestor_of_share(Path::new("/opt")));
    }

    #[test]
    fn test_path_visibility() {
        let registry = ShareRegistry::new();
        registry.add_share(make_share("/home/user/project", ShareMode::ReadWrite));

        // Share path itself
        assert!(registry.is_path_visible(Path::new("/home/user/project")));

        // Paths under the share
        assert!(registry.is_path_visible(Path::new("/home/user/project/src")));

        // Ancestors (for navigation)
        assert!(registry.is_path_visible(Path::new("/home")));
        assert!(registry.is_path_visible(Path::new("/home/user")));

        // Unrelated paths
        assert!(!registry.is_path_visible(Path::new("/opt")));
        assert!(!registry.is_path_visible(Path::new("/home/other")));
    }

    #[test]
    fn test_list_shares() {
        let registry = ShareRegistry::new();
        registry.add_share(make_share("/home/user", ShareMode::ReadOnly));
        registry.add_share(make_share("/opt/data", ShareMode::ReadWrite));

        let shares = registry.list_shares();
        assert_eq!(shares.len(), 2);
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let registry = Arc::new(ShareRegistry::new());

        // Spawn multiple threads that read and write concurrently
        let mut handles = vec![];

        for i in 0..10 {
            let reg = Arc::clone(&registry);
            handles.push(thread::spawn(move || {
                let path = format!("/share{}", i);
                reg.add_share(make_share(&path, ShareMode::ReadOnly));
            }));
        }

        for _ in 0..10 {
            let reg = Arc::clone(&registry);
            handles.push(thread::spawn(move || {
                let _ = reg.list_shares();
                let _ = reg.find_share(Path::new("/share5/file.txt"));
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // All shares should be present
        assert_eq!(registry.len(), 10);
    }
}
