// Copyright 2026 The Virtiofs Project Developers
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

//! Share definitions for dynamic path sharing.
//!
//! A [`Share`] represents a host path that is shared with the guest VM,
//! along with its access mode (read-only or read-write).

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::str::FromStr;

/// Access mode for a shared path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ShareMode {
    /// Read-only access
    ReadOnly,
    /// Read-write access
    ReadWrite,
}

impl ShareMode {
    /// Returns true if this mode allows write operations.
    pub fn is_writable(&self) -> bool {
        matches!(self, ShareMode::ReadWrite)
    }

    /// Returns the short form string representation.
    pub fn as_short_str(&self) -> &'static str {
        match self {
            ShareMode::ReadOnly => "ro",
            ShareMode::ReadWrite => "rw",
        }
    }
}

impl FromStr for ShareMode {
    type Err = ShareParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "ro" | "readonly" | "read-only" => Ok(ShareMode::ReadOnly),
            "rw" | "readwrite" | "read-write" => Ok(ShareMode::ReadWrite),
            _ => Err(ShareParseError::InvalidMode(s.to_string())),
        }
    }
}

impl std::fmt::Display for ShareMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShareMode::ReadOnly => write!(f, "readonly"),
            ShareMode::ReadWrite => write!(f, "readwrite"),
        }
    }
}

/// A shared path with its access mode.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Share {
    /// The canonical host path being shared.
    path: PathBuf,
    /// Access mode for this share.
    mode: ShareMode,
}

impl Share {
    /// Create a new share with the given path and mode.
    ///
    /// The path will be canonicalized.
    pub fn new(path: impl AsRef<Path>, mode: ShareMode) -> std::io::Result<Self> {
        let canonical = std::fs::canonicalize(path.as_ref())?;
        Ok(Share {
            path: canonical,
            mode,
        })
    }

    /// Create a share without canonicalizing the path.
    ///
    /// Use with caution - the path should already be canonical.
    pub fn new_unchecked(path: PathBuf, mode: ShareMode) -> Self {
        Share { path, mode }
    }

    /// Returns the shared path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the access mode.
    pub fn mode(&self) -> ShareMode {
        self.mode
    }

    /// Returns the number of path components.
    pub fn depth(&self) -> usize {
        self.path.components().count()
    }

    /// Returns true if the given path is under this share.
    pub fn contains(&self, path: &Path) -> bool {
        path.starts_with(&self.path)
    }

    /// Returns the relative path from this share's root to the given path.
    ///
    /// Returns `None` if the path is not under this share.
    pub fn relative_path<'a>(&self, path: &'a Path) -> Option<&'a Path> {
        path.strip_prefix(&self.path).ok()
    }
}

impl FromStr for Share {
    type Err = ShareParseError;

    /// Parse a share from the format `path:mode` (e.g., `/home/user:ro`).
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Find the last colon to split path and mode
        let colon_pos = s
            .rfind(':')
            .ok_or_else(|| ShareParseError::MissingMode(s.to_string()))?;

        let path_str = &s[..colon_pos];
        let mode_str = &s[colon_pos + 1..];

        if path_str.is_empty() {
            return Err(ShareParseError::EmptyPath);
        }

        let mode = mode_str.parse()?;
        let path = PathBuf::from(path_str);

        // Verify the path exists
        if !path.exists() {
            return Err(ShareParseError::PathNotFound(path));
        }

        Share::new(path, mode).map_err(|e| ShareParseError::IoError(e.to_string()))
    }
}

impl std::fmt::Display for Share {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.path.display(), self.mode.as_short_str())
    }
}

/// Errors that can occur when parsing a share specification.
#[derive(Debug, Clone)]
pub enum ShareParseError {
    /// The share specification is missing the mode (e.g., `:ro` or `:rw`).
    MissingMode(String),
    /// The mode is not recognized.
    InvalidMode(String),
    /// The path is empty.
    EmptyPath,
    /// The path does not exist.
    PathNotFound(PathBuf),
    /// An I/O error occurred while processing the path.
    IoError(String),
}

impl std::fmt::Display for ShareParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShareParseError::MissingMode(s) => {
                write!(f, "share specification '{s}' is missing mode (use path:ro or path:rw)")
            }
            ShareParseError::InvalidMode(s) => {
                write!(f, "invalid share mode '{s}' (expected 'ro' or 'rw')")
            }
            ShareParseError::EmptyPath => write!(f, "share path cannot be empty"),
            ShareParseError::PathNotFound(p) => {
                write!(f, "share path '{}' does not exist", p.display())
            }
            ShareParseError::IoError(e) => write!(f, "I/O error processing share: {e}"),
        }
    }
}

impl std::error::Error for ShareParseError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_share_mode_parse() {
        assert_eq!("ro".parse::<ShareMode>().unwrap(), ShareMode::ReadOnly);
        assert_eq!("RO".parse::<ShareMode>().unwrap(), ShareMode::ReadOnly);
        assert_eq!("readonly".parse::<ShareMode>().unwrap(), ShareMode::ReadOnly);
        assert_eq!("rw".parse::<ShareMode>().unwrap(), ShareMode::ReadWrite);
        assert_eq!("RW".parse::<ShareMode>().unwrap(), ShareMode::ReadWrite);
        assert_eq!("readwrite".parse::<ShareMode>().unwrap(), ShareMode::ReadWrite);
        assert!("invalid".parse::<ShareMode>().is_err());
    }

    #[test]
    fn test_share_mode_display() {
        assert_eq!(ShareMode::ReadOnly.to_string(), "readonly");
        assert_eq!(ShareMode::ReadWrite.to_string(), "readwrite");
        assert_eq!(ShareMode::ReadOnly.as_short_str(), "ro");
        assert_eq!(ShareMode::ReadWrite.as_short_str(), "rw");
    }

    #[test]
    fn test_share_contains() {
        let share = Share::new_unchecked(PathBuf::from("/home/user"), ShareMode::ReadOnly);
        assert!(share.contains(Path::new("/home/user")));
        assert!(share.contains(Path::new("/home/user/documents")));
        assert!(!share.contains(Path::new("/home/other")));
        assert!(!share.contains(Path::new("/home")));
    }

    #[test]
    fn test_share_relative_path() {
        let share = Share::new_unchecked(PathBuf::from("/home/user"), ShareMode::ReadOnly);
        assert_eq!(
            share.relative_path(Path::new("/home/user/documents")),
            Some(Path::new("documents"))
        );
        assert_eq!(
            share.relative_path(Path::new("/home/user")),
            Some(Path::new(""))
        );
        assert_eq!(share.relative_path(Path::new("/home/other")), None);
    }

    #[test]
    fn test_share_depth() {
        let share = Share::new_unchecked(PathBuf::from("/home/user"), ShareMode::ReadOnly);
        assert_eq!(share.depth(), 3); // /, home, user

        let share2 = Share::new_unchecked(PathBuf::from("/"), ShareMode::ReadOnly);
        assert_eq!(share2.depth(), 1);
    }
}
