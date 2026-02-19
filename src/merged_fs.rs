// Copyright 2026 The Virtiofs Project Developers
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

//! Merged filesystem implementation for dynamic path sharing.
//!
//! [`MergedPathFs`] wraps a [`PassthroughFs`] and adds:
//! - Visibility filtering based on shares (paths outside shares return ENOENT)
//! - Permission enforcement (write operations fail on read-only shares)
//! - Path specificity rules (more specific shares override less specific ones)

use crate::filesystem::{
    Context, DirEntry, DirectoryIterator, Entry, Extensions, FileSystem, FsOptions, GetxattrReply,
    ListxattrReply, OpenOptions, SerializableFileSystem, SetattrValid, SetxattrFlags,
    ZeroCopyReader, ZeroCopyWriter,
};
use crate::fuse;
use crate::passthrough::{Config, PassthroughFs};
use crate::share::{Share, ShareMode};
use crate::share_registry::ShareRegistry;
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, RwLock};
use std::time::Duration;

/// Inode type from passthrough.
pub type Inode = u64;

/// Handle type from passthrough.
pub type Handle = u64;

// Helper functions for permission errors
fn enoent() -> io::Error {
    io::Error::from_raw_os_error(libc::ENOENT)
}

fn erofs() -> io::Error {
    io::Error::from_raw_os_error(libc::EROFS)
}

fn eacces() -> io::Error {
    io::Error::from_raw_os_error(libc::EACCES)
}

/// Cached directory entry with owned data.
#[derive(Clone)]
struct CachedDirEntry {
    ino: libc::ino64_t,
    type_: u32,
    name: CString,
}

/// Cache of filtered directory entries for an open directory handle.
struct FilteredDirCache {
    entries: Vec<CachedDirEntry>,
}

/// A filtered directory iterator that only yields visible entries.
///
/// This wraps a collected set of directory entries with virtual sequential
/// offsets to work correctly with FUSE's offset-based pagination.
pub struct FilteredReadDir {
    entries: Vec<CachedDirEntry>,
    index: usize,
}

impl FilteredReadDir {
    /// Create a new filtered directory iterator from cached entries.
    ///
    /// The `offset` parameter is the FUSE offset from the last returned entry.
    /// We use it as an index into our cached entries (offset 0 = start, offset N = start at index N).
    fn from_cache(entries: Vec<CachedDirEntry>, offset: u64) -> Self {
        let index = offset as usize;
        FilteredReadDir { entries, index }
    }
}

impl DirectoryIterator for FilteredReadDir {
    fn next(&mut self) -> Option<DirEntry<'_>> {
        if self.index >= self.entries.len() {
            return None;
        }

        let entry = &self.entries[self.index];
        // Virtual sequential offset: index + 1 (so offset 0 means "start from beginning")
        let virtual_offset = (self.index + 1) as u64;
        self.index += 1;

        Some(DirEntry {
            ino: entry.ino,
            offset: virtual_offset,
            type_: entry.type_,
            name: entry.name.as_c_str(),
        })
    }
}

/// A filesystem that merges multiple shared paths with per-path permissions.
///
/// This wrapper around [`PassthroughFs`] adds:
/// - Visibility filtering: paths not under any share return ENOENT
/// - Permission enforcement: write operations fail on read-only shares
/// - Path specificity: more specific shares override less specific ones
pub struct MergedPathFs {
    /// The underlying passthrough filesystem.
    inner: PassthroughFs,
    /// Registry of shared paths and their permissions.
    registry: Arc<ShareRegistry>,
    /// Mapping from inode to its absolute path.
    /// Populated during lookup() and cleaned up during forget().
    inode_paths: RwLock<HashMap<Inode, PathBuf>>,
    /// Cache of filtered directory contents per directory handle.
    /// Populated during opendir() and cleaned up during releasedir().
    dir_cache: RwLock<HashMap<Handle, FilteredDirCache>>,
    /// Prefix for all paths in the guest view (e.g., "/mnt/host").
    /// Will be used when path translation is implemented.
    #[allow(dead_code)]
    mount_prefix: Option<PathBuf>,
    /// The root directory path (stored separately since inner.cfg is private).
    root_dir: PathBuf,
}

impl MergedPathFs {
    /// Create a new `MergedPathFs` with the given shares.
    ///
    /// The `cfg.root_dir` should be set to "/" or a common ancestor of all shares
    /// when using `--sandbox=none`.
    pub fn new(
        cfg: Config,
        shares: impl IntoIterator<Item = Share>,
        mount_prefix: Option<PathBuf>,
    ) -> io::Result<Self> {
        let root_dir = PathBuf::from(&cfg.root_dir);
        let inner = PassthroughFs::new(cfg)?;
        let registry = Arc::new(ShareRegistry::with_shares(shares));

        // Initialize inode_paths with the root inode
        let mut inode_paths = HashMap::new();
        inode_paths.insert(fuse::ROOT_ID, root_dir.clone());

        Ok(MergedPathFs {
            inner,
            registry,
            inode_paths: RwLock::new(inode_paths),
            dir_cache: RwLock::new(HashMap::new()),
            mount_prefix,
            root_dir,
        })
    }

    /// Get a reference to the share registry.
    ///
    /// This can be used to dynamically add or remove shares.
    pub fn registry(&self) -> &Arc<ShareRegistry> {
        &self.registry
    }

    /// Get the configuration root directory.
    pub fn root_dir(&self) -> &Path {
        &self.root_dir
    }

    /// Get the path for an inode.
    ///
    /// Returns `None` if the inode is not in our tracking map.
    fn get_inode_path(&self, inode: Inode) -> Option<PathBuf> {
        self.inode_paths.read().unwrap().get(&inode).cloned()
    }

    /// Record the path for an inode.
    fn set_inode_path(&self, inode: Inode, path: PathBuf) {
        self.inode_paths.write().unwrap().insert(inode, path);
    }

    /// Remove an inode from the path tracking map.
    ///
    /// Currently unused but kept for potential future use in forget() cleanup.
    #[allow(dead_code)]
    fn remove_inode_path(&self, inode: Inode) {
        self.inode_paths.write().unwrap().remove(&inode);
    }

    /// Check if read access (visibility) is allowed for the given inode.
    ///
    /// Returns `Ok(path)` if visible, or `Err(ENOENT)` if not visible.
    fn check_inode_visible(&self, inode: Inode) -> io::Result<PathBuf> {
        let path = self.get_inode_path(inode).ok_or_else(enoent)?;
        if self.registry.is_path_visible(&path) {
            Ok(path)
        } else {
            Err(enoent())
        }
    }

    /// Check if write access is allowed for the given inode.
    ///
    /// Returns `Ok(path)` if writable, `Err(EROFS)` if read-only, or `Err(ENOENT)` if not visible.
    fn check_inode_writable(&self, inode: Inode) -> io::Result<PathBuf> {
        let path = self.get_inode_path(inode).ok_or_else(enoent)?;
        match self.registry.get_permission(&path) {
            Some(ShareMode::ReadWrite) => Ok(path),
            Some(ShareMode::ReadOnly) => Err(erofs()),
            None => {
                // Check if it's an ancestor (which are read-only for navigation)
                if self.registry.is_ancestor_of_share(&path) {
                    Err(erofs())
                } else {
                    Err(enoent())
                }
            }
        }
    }

    /// Check if creating a child entry under a parent inode is allowed.
    ///
    /// Unlike `check_inode_writable`, this checks the *child* path's permission,
    /// which handles the case where the parent is an ancestor directory but the
    /// child falls under an RW share (e.g., parent=/home/user, child=.claude.lock,
    /// share=/home/user/.claude.lock:rw).
    fn check_create_allowed(&self, parent: Inode, name: &CStr) -> io::Result<PathBuf> {
        let parent_path = self.get_inode_path(parent).ok_or_else(enoent)?;
        let name_str = name.to_str().map_err(|_| enoent())?;
        let child_path = parent_path.join(name_str);

        match self.registry.get_permission(&child_path) {
            Some(ShareMode::ReadWrite) => Ok(parent_path),
            Some(ShareMode::ReadOnly) => Err(erofs()),
            None => {
                // Child isn't directly under a share — fall back to parent check
                self.check_inode_writable(parent)
            }
        }
    }

    /// Check if a path should be visible based on the share configuration.
    fn is_path_visible(&self, path: &Path) -> bool {
        self.registry.is_path_visible(path)
    }

    /// Check if a path is writable based on the share configuration.
    fn is_path_writable(&self, path: &Path) -> bool {
        matches!(
            self.registry.get_permission(path),
            Some(ShareMode::ReadWrite)
        )
    }

    /// Internal helper to check open flags and determine if write access is needed.
    fn open_needs_write(&self, flags: u32) -> bool {
        let cflags: libc::c_int = flags as libc::c_int;

        // O_PATH doesn't need write access
        if cflags & libc::O_PATH != 0 {
            return false;
        }

        // Check access mode
        let accmode = cflags & libc::O_ACCMODE;
        accmode == libc::O_WRONLY || accmode == libc::O_RDWR
    }
}

// Macros for delegating methods to inner PassthroughFs
macro_rules! delegate_allow {
    {
        $(
            fn $name:ident$(<$($gen_name:ident: $gen_trait:path),*>)?(
                &self
                $(, $($par_name:ident: $par_type:ty),*)?
                $(,)?
            )$( -> $ret:ty)?;
        )*
    } => {
        $(
            fn $name$(<$($gen_name: $gen_trait),*>)?(
                &self
                $(, $($par_name: $par_type),*)?
            )$( -> $ret)? {
                self.inner.$name($($($par_name),*)?)
            }
        )*
    }
}

impl FileSystem for MergedPathFs {
    type Inode = Inode;
    type Handle = Handle;
    type DirIter = FilteredReadDir;

    // Initialize - allow through
    delegate_allow! {
        fn init(&self, capable: FsOptions) -> io::Result<FsOptions>;
        fn destroy(&self);
    }

    // Lookup - we need to check visibility and track inode paths
    fn lookup(&self, ctx: Context, parent: Self::Inode, name: &CStr) -> io::Result<Entry> {
        // Get the parent path first
        let parent_path = self.get_inode_path(parent).ok_or_else(|| {
            log::debug!("lookup: parent inode {} not in path map", parent);
            enoent()
        })?;

        // Build the full path for this entry
        let name_str = name.to_str().map_err(|_| enoent())?;
        let full_path = parent_path.join(name_str);

        // Check if this path should be visible
        let visible = self.is_path_visible(&full_path);
        log::debug!("lookup: {:?} visible={}", full_path, visible);

        if !visible {
            return Err(enoent());
        }

        // Delegate lookup to inner
        let entry = self.inner.lookup(ctx, parent, name)?;

        // Track this inode's path
        self.set_inode_path(entry.inode, full_path);

        Ok(entry)
    }

    // Forget - clean up our inode path tracking
    fn forget(&self, ctx: Context, inode: Self::Inode, count: u64) {
        // Note: We don't remove from inode_paths here because the kernel
        // may still have references. The path tracking is eventually consistent
        // and extra entries are harmless.
        self.inner.forget(ctx, inode, count);
    }

    fn batch_forget(&self, ctx: Context, requests: Vec<(Self::Inode, u64)>) {
        self.inner.batch_forget(ctx, requests);
    }

    // Read operations - allow through, visibility is checked at lookup
    delegate_allow! {
        fn getattr(&self,
            ctx: Context,
            inode: Self::Inode,
            handle: Option<Self::Handle>,
        ) -> io::Result<(fuse::Attr, Duration)>;
        fn readlink(&self, ctx: Context, inode: Self::Inode) -> io::Result<Vec<u8>>;
        fn read<W: ZeroCopyWriter>(
            &self,
            ctx: Context,
            inode: Self::Inode,
            handle: Self::Handle,
            w: W,
            size: u32,
            offset: u64,
            lock_owner: Option<u64>,
            flags: u32,
        ) -> io::Result<usize>;
        fn flush(
            &self,
            ctx: Context,
            inode: Self::Inode,
            handle: Self::Handle,
            lock_owner: u64,
        ) -> io::Result<()>;
        fn fsync(
            &self,
            ctx: Context,
            inode: Self::Inode,
            datasync: bool,
            handle: Self::Handle,
        ) -> io::Result<()>;
        fn release(
            &self,
            ctx: Context,
            inode: Self::Inode,
            flags: u32,
            handle: Self::Handle,
            flush: bool,
            flock_release: bool,
            lock_owner: Option<u64>,
        ) -> io::Result<()>;
        fn statfs(&self, ctx: Context, inode: Self::Inode) -> io::Result<libc::statvfs64>;
        fn getxattr(
            &self,
            ctx: Context,
            inode: Self::Inode,
            name: &CStr,
            size: u32,
        ) -> io::Result<GetxattrReply>;
        fn listxattr(
            &self,
            ctx: Context,
            inode: Self::Inode,
            size: u32,
        ) -> io::Result<ListxattrReply>;
        fn fsyncdir(
            &self,
            ctx: Context,
            inode: Self::Inode,
            datasync: bool,
            handle: Self::Handle,
        ) -> io::Result<()>;
        fn lseek(
            &self,
            ctx: Context,
            inode: Self::Inode,
            handle: Self::Handle,
            offset: u64,
            whence: u32,
        ) -> io::Result<u64>;
        fn syncfs(&self, ctx: Context, inode: Self::Inode) -> io::Result<()>;
    }

    // Readdir - return filtered entries from our cache with virtual offsets
    //
    // We cache filtered directory contents at opendir() time and return them
    // here with sequential virtual offsets. This solves the FUSE pagination
    // problem where filtering with original offsets causes infinite loops.
    fn readdir(
        &self,
        _ctx: Context,
        inode: Self::Inode,
        handle: Self::Handle,
        _size: u32,
        offset: u64,
    ) -> io::Result<Self::DirIter> {
        self.check_inode_visible(inode)?;

        // Look up cached entries for this handle
        let cache = self.dir_cache.read().unwrap();
        let cached = cache.get(&handle).ok_or_else(|| {
            log::warn!("readdir: no cache entry for handle {}", handle);
            enoent()
        })?;

        // Clone the entries since we need to return an owned iterator
        let entries = cached.entries.clone();
        drop(cache);

        log::debug!(
            "readdir inode={} handle={} offset={}: returning from cache ({} entries)",
            inode,
            handle,
            offset,
            entries.len()
        );

        Ok(FilteredReadDir::from_cache(entries, offset))
    }

    // Open with permission checking
    fn open(
        &self,
        ctx: Context,
        inode: Self::Inode,
        kill_priv: bool,
        flags: u32,
    ) -> io::Result<(Option<Self::Handle>, OpenOptions)> {
        // Check if write access is requested
        if self.open_needs_write(flags) {
            self.check_inode_writable(inode)?;
        } else {
            self.check_inode_visible(inode)?;
        }
        self.inner.open(ctx, inode, kill_priv, flags)
    }

    fn opendir(
        &self,
        ctx: Context,
        inode: Self::Inode,
        flags: u32,
    ) -> io::Result<(Option<Self::Handle>, OpenOptions)> {
        let parent_path = self.check_inode_visible(inode)?;

        // Open the directory in the underlying filesystem
        let (handle_opt, opts) = self.inner.opendir(ctx, inode, flags)?;

        // If we got a handle, cache filtered directory contents
        if let Some(handle) = handle_opt {
            // Read entire directory from inner filesystem
            // Use a large buffer size and offset 0 to start
            const READDIR_BUF_SIZE: u32 = 1024 * 1024; // 1MB buffer
            let mut filtered_entries = Vec::new();
            let mut offset = 0u64;
            let mut total_count = 0u32;

            loop {
                let mut iter = self.inner.readdir(ctx, inode, handle, READDIR_BUF_SIZE, offset)?;
                let mut got_entries = false;

                while let Some(entry) = iter.next() {
                    got_entries = true;
                    total_count += 1;
                    offset = entry.offset; // Track last offset for continuation
                    let name_bytes = entry.name.to_bytes();

                    // Always include . and ..
                    if name_bytes == b"." || name_bytes == b".." {
                        if let Ok(name) = CString::new(name_bytes) {
                            filtered_entries.push(CachedDirEntry {
                                ino: entry.ino,
                                type_: entry.type_,
                                name,
                            });
                        }
                        continue;
                    }

                    // Check visibility for other entries
                    if let Ok(name_str) = entry.name.to_str() {
                        let full_path = parent_path.join(name_str);
                        let visible = self.registry.is_path_visible(&full_path);

                        if visible {
                            if let Ok(name) = CString::new(name_bytes) {
                                filtered_entries.push(CachedDirEntry {
                                    ino: entry.ino,
                                    type_: entry.type_,
                                    name,
                                });
                            }
                        }
                    }
                }

                if !got_entries {
                    break;
                }
            }

            log::debug!(
                "opendir inode={} handle={}: {} total entries, {} after filtering",
                inode,
                handle,
                total_count,
                filtered_entries.len()
            );

            // Cache the filtered entries
            self.dir_cache.write().unwrap().insert(
                handle,
                FilteredDirCache {
                    entries: filtered_entries,
                },
            );
        }

        Ok((handle_opt, opts))
    }

    fn releasedir(
        &self,
        ctx: Context,
        inode: Self::Inode,
        flags: u32,
        handle: Self::Handle,
    ) -> io::Result<()> {
        // Clean up cache entry for this handle
        if self.dir_cache.write().unwrap().remove(&handle).is_some() {
            log::debug!("releasedir: cleaned up cache for handle {}", handle);
        }

        // Delegate to inner
        self.inner.releasedir(ctx, inode, flags, handle)
    }

    fn access(&self, ctx: Context, inode: Self::Inode, mask: u32) -> io::Result<()> {
        let path = self.check_inode_visible(inode)?;

        // Check write access if W_OK is requested
        if mask & (libc::W_OK as u32) != 0 && !self.is_path_writable(&path) {
            return Err(eacces());
        }
        self.inner.access(ctx, inode, mask)
    }

    // Write operations - check permissions before delegating
    fn setattr(
        &self,
        ctx: Context,
        inode: Self::Inode,
        attr: fuse::SetattrIn,
        handle: Option<Self::Handle>,
        valid: SetattrValid,
    ) -> io::Result<(fuse::Attr, Duration)> {
        self.check_inode_writable(inode)?;
        self.inner.setattr(ctx, inode, attr, handle, valid)
    }

    fn symlink(
        &self,
        ctx: Context,
        linkname: &CStr,
        parent: Self::Inode,
        name: &CStr,
        extensions: Extensions,
    ) -> io::Result<Entry> {
        let parent_path = self.check_create_allowed(parent, name)?;
        let name_str = name.to_str().map_err(|_| enoent())?;
        let new_path = parent_path.join(name_str);

        let entry = self.inner.symlink(ctx, linkname, parent, name, extensions)?;
        self.set_inode_path(entry.inode, new_path);
        Ok(entry)
    }

    fn mknod(
        &self,
        ctx: Context,
        parent: Self::Inode,
        name: &CStr,
        mode: u32,
        rdev: u32,
        umask: u32,
        extensions: Extensions,
    ) -> io::Result<Entry> {
        let parent_path = self.check_create_allowed(parent, name)?;
        let name_str = name.to_str().map_err(|_| enoent())?;
        let new_path = parent_path.join(name_str);

        let entry = self.inner.mknod(ctx, parent, name, mode, rdev, umask, extensions)?;
        self.set_inode_path(entry.inode, new_path);
        Ok(entry)
    }

    fn mkdir(
        &self,
        ctx: Context,
        parent: Self::Inode,
        name: &CStr,
        mode: u32,
        umask: u32,
        extensions: Extensions,
    ) -> io::Result<Entry> {
        let parent_path = self.check_create_allowed(parent, name)?;
        let name_str = name.to_str().map_err(|_| enoent())?;
        let new_path = parent_path.join(name_str);

        let entry = self.inner.mkdir(ctx, parent, name, mode, umask, extensions)?;
        self.set_inode_path(entry.inode, new_path);
        Ok(entry)
    }

    fn unlink(&self, ctx: Context, parent: Self::Inode, name: &CStr) -> io::Result<()> {
        self.check_create_allowed(parent, name)?;
        self.inner.unlink(ctx, parent, name)
    }

    fn rmdir(&self, ctx: Context, parent: Self::Inode, name: &CStr) -> io::Result<()> {
        self.check_create_allowed(parent, name)?;
        self.inner.rmdir(ctx, parent, name)
    }

    fn rename(
        &self,
        ctx: Context,
        olddir: Self::Inode,
        oldname: &CStr,
        newdir: Self::Inode,
        newname: &CStr,
        flags: u32,
    ) -> io::Result<()> {
        // Both source and destination child paths must be writable
        self.check_create_allowed(olddir, oldname)?;
        self.check_create_allowed(newdir, newname)?;
        self.inner.rename(ctx, olddir, oldname, newdir, newname, flags)
    }

    fn link(
        &self,
        ctx: Context,
        inode: Self::Inode,
        newparent: Self::Inode,
        newname: &CStr,
    ) -> io::Result<Entry> {
        // Source must be visible, destination child path must be writable
        self.check_inode_visible(inode)?;
        let parent_path = self.check_create_allowed(newparent, newname)?;
        let name_str = newname.to_str().map_err(|_| enoent())?;
        let new_path = parent_path.join(name_str);

        let entry = self.inner.link(ctx, inode, newparent, newname)?;
        self.set_inode_path(entry.inode, new_path);
        Ok(entry)
    }

    fn write<R: ZeroCopyReader>(
        &self,
        ctx: Context,
        inode: Self::Inode,
        handle: Self::Handle,
        r: R,
        size: u32,
        offset: u64,
        lock_owner: Option<u64>,
        delayed_write: bool,
        kill_priv: bool,
        flags: u32,
    ) -> io::Result<usize> {
        self.check_inode_writable(inode)?;
        self.inner.write(ctx, inode, handle, r, size, offset, lock_owner, delayed_write, kill_priv, flags)
    }

    fn create(
        &self,
        ctx: Context,
        parent: Self::Inode,
        name: &CStr,
        mode: u32,
        kill_priv: bool,
        flags: u32,
        umask: u32,
        extensions: Extensions,
    ) -> io::Result<(Entry, Option<Self::Handle>, OpenOptions)> {
        let parent_path = self.check_create_allowed(parent, name)?;
        let name_str = name.to_str().map_err(|_| enoent())?;
        let new_path = parent_path.join(name_str);

        let (entry, handle, opts) = self.inner.create(ctx, parent, name, mode, kill_priv, flags, umask, extensions)?;
        self.set_inode_path(entry.inode, new_path);
        Ok((entry, handle, opts))
    }

    fn fallocate(
        &self,
        ctx: Context,
        inode: Self::Inode,
        handle: Self::Handle,
        mode: u32,
        offset: u64,
        length: u64,
    ) -> io::Result<()> {
        self.check_inode_writable(inode)?;
        self.inner.fallocate(ctx, inode, handle, mode, offset, length)
    }

    fn setxattr(
        &self,
        ctx: Context,
        inode: Self::Inode,
        name: &CStr,
        value: &[u8],
        flags: u32,
        extra_flags: SetxattrFlags,
    ) -> io::Result<()> {
        self.check_inode_writable(inode)?;
        self.inner.setxattr(ctx, inode, name, value, flags, extra_flags)
    }

    fn removexattr(&self, ctx: Context, inode: Self::Inode, name: &CStr) -> io::Result<()> {
        self.check_inode_writable(inode)?;
        self.inner.removexattr(ctx, inode, name)
    }

    fn copyfilerange(
        &self,
        ctx: Context,
        inode_in: Self::Inode,
        handle_in: Self::Handle,
        offset_in: u64,
        inode_out: Self::Inode,
        handle_out: Self::Handle,
        offset_out: u64,
        len: u64,
        flags: u64,
    ) -> io::Result<usize> {
        // Source must be visible, destination must be writable
        self.check_inode_visible(inode_in)?;
        self.check_inode_writable(inode_out)?;
        self.inner.copyfilerange(ctx, inode_in, handle_in, offset_in, inode_out, handle_out, offset_out, len, flags)
    }
}

impl SerializableFileSystem for MergedPathFs {
    fn prepare_serialization(&self, cancel: Arc<AtomicBool>) {
        self.inner.prepare_serialization(cancel)
    }

    fn serialize(&self, state_pipe: File) -> io::Result<()> {
        self.inner.serialize(state_pipe)
    }

    fn deserialize_and_apply(&self, state_pipe: File) -> io::Result<()> {
        self.inner.deserialize_and_apply(state_pipe)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_share(path: &str, mode: ShareMode) -> Share {
        Share::new_unchecked(PathBuf::from(path), mode)
    }

    #[test]
    fn test_check_path_access_under_share() {
        // This test verifies the path access logic
        let registry = ShareRegistry::with_shares(vec![
            make_share("/home/user", ShareMode::ReadOnly),
            make_share("/home/user/project", ShareMode::ReadWrite),
        ]);

        // Path under read-only share
        assert_eq!(
            registry.get_permission(Path::new("/home/user/documents")),
            Some(ShareMode::ReadOnly)
        );

        // Path under read-write share (more specific)
        assert_eq!(
            registry.get_permission(Path::new("/home/user/project/src")),
            Some(ShareMode::ReadWrite)
        );

        // Path not under any share
        assert_eq!(registry.get_permission(Path::new("/etc")), None);
    }

    #[test]
    fn test_ancestor_visibility() {
        let registry = ShareRegistry::with_shares(vec![make_share(
            "/home/user/project",
            ShareMode::ReadWrite,
        )]);

        // Ancestors should be visible for navigation
        assert!(registry.is_ancestor_of_share(Path::new("/home")));
        assert!(registry.is_ancestor_of_share(Path::new("/home/user")));

        // But not the share itself
        assert!(!registry.is_ancestor_of_share(Path::new("/home/user/project")));
    }

    #[test]
    fn test_visibility_combined() {
        // Test the combined visibility check (is_path_visible)
        let registry = ShareRegistry::with_shares(vec![
            make_share("/home/user/project", ShareMode::ReadWrite),
            make_share("/var/data", ShareMode::ReadOnly),
        ]);

        // Paths under shares are visible
        assert!(registry.is_path_visible(Path::new("/home/user/project")));
        assert!(registry.is_path_visible(Path::new("/home/user/project/src/main.rs")));
        assert!(registry.is_path_visible(Path::new("/var/data")));
        assert!(registry.is_path_visible(Path::new("/var/data/file.txt")));

        // Ancestors of shares are visible for navigation
        assert!(registry.is_path_visible(Path::new("/home")));
        assert!(registry.is_path_visible(Path::new("/home/user")));
        assert!(registry.is_path_visible(Path::new("/var")));

        // Root is an ancestor of all shares
        assert!(registry.is_path_visible(Path::new("/")));

        // Unrelated paths are not visible
        assert!(!registry.is_path_visible(Path::new("/etc")));
        assert!(!registry.is_path_visible(Path::new("/home/other")));
        assert!(!registry.is_path_visible(Path::new("/tmp")));
    }

    #[test]
    fn test_write_permission_rw_share() {
        let registry = ShareRegistry::with_shares(vec![
            make_share("/data/writable", ShareMode::ReadWrite),
        ]);

        // Paths under rw share should be writable
        assert_eq!(
            registry.get_permission(Path::new("/data/writable")),
            Some(ShareMode::ReadWrite)
        );
        assert_eq!(
            registry.get_permission(Path::new("/data/writable/subdir/file")),
            Some(ShareMode::ReadWrite)
        );
    }

    #[test]
    fn test_write_permission_ro_share() {
        let registry = ShareRegistry::with_shares(vec![
            make_share("/data/readonly", ShareMode::ReadOnly),
        ]);

        // Paths under ro share should not be writable
        assert_eq!(
            registry.get_permission(Path::new("/data/readonly")),
            Some(ShareMode::ReadOnly)
        );
        assert_eq!(
            registry.get_permission(Path::new("/data/readonly/file")),
            Some(ShareMode::ReadOnly)
        );
    }

    #[test]
    fn test_write_permission_specificity() {
        // More specific share overrides less specific
        let registry = ShareRegistry::with_shares(vec![
            make_share("/data", ShareMode::ReadOnly),
            make_share("/data/writable", ShareMode::ReadWrite),
        ]);

        // Parent is read-only
        assert_eq!(
            registry.get_permission(Path::new("/data/other")),
            Some(ShareMode::ReadOnly)
        );

        // More specific path is read-write
        assert_eq!(
            registry.get_permission(Path::new("/data/writable/file")),
            Some(ShareMode::ReadWrite)
        );
    }

    #[test]
    fn test_write_permission_outside_share() {
        let registry = ShareRegistry::with_shares(vec![
            make_share("/data/shared", ShareMode::ReadWrite),
        ]);

        // Paths outside any share have no permission
        assert_eq!(registry.get_permission(Path::new("/etc")), None);
        assert_eq!(registry.get_permission(Path::new("/tmp")), None);

        // Ancestor paths are not directly in the share
        assert_eq!(registry.get_permission(Path::new("/data")), None);
    }

    /// Helper to filter entries like opendir does (simulates caching logic)
    fn filter_entries(
        entries: Vec<(libc::ino64_t, u32, &'static [u8])>,
        parent_path: &Path,
        registry: &ShareRegistry,
    ) -> FilteredReadDir {
        let mut filtered = Vec::new();
        for (ino, type_, name_bytes) in entries {
            // Remove null terminator for comparison
            let name_slice = if name_bytes.last() == Some(&0) {
                &name_bytes[..name_bytes.len() - 1]
            } else {
                name_bytes
            };

            // Always include . and ..
            if name_slice == b"." || name_slice == b".." {
                if let Ok(name) = CString::new(name_slice) {
                    filtered.push(CachedDirEntry { ino, type_, name });
                }
                continue;
            }

            // Check visibility
            if let Ok(name_str) = std::str::from_utf8(name_slice) {
                let full_path = parent_path.join(name_str);
                if registry.is_path_visible(&full_path) {
                    if let Ok(name) = CString::new(name_slice) {
                        filtered.push(CachedDirEntry { ino, type_, name });
                    }
                }
            }
        }
        FilteredReadDir::from_cache(filtered, 0)
    }

    #[test]
    fn test_filtered_readdir_includes_dot_entries() {
        let entries = vec![
            (1, 4, b".\0" as &[u8]),
            (2, 4, b"..\0"),
            (3, 4, b"visible\0"),
            (4, 4, b"hidden\0"),
        ];

        let registry = ShareRegistry::with_shares(vec![
            make_share("/parent/visible", ShareMode::ReadWrite),
        ]);

        let parent_path = Path::new("/parent");
        let mut filtered = filter_entries(entries, parent_path, &registry);

        // . and .. should always be included
        let entry1 = filtered.next().unwrap();
        assert_eq!(entry1.name.to_bytes(), b".");

        let entry2 = filtered.next().unwrap();
        assert_eq!(entry2.name.to_bytes(), b"..");

        // visible should be included (it's under a share)
        let entry3 = filtered.next().unwrap();
        assert_eq!(entry3.name.to_bytes(), b"visible");

        // hidden should be filtered out (not under any share)
        assert!(filtered.next().is_none());
    }

    #[test]
    fn test_filtered_readdir_filters_invisible() {
        let entries = vec![
            (1, 4, b".\0" as &[u8]),
            (2, 4, b"..\0"),
            (3, 4, b"shared\0"),
            (4, 4, b"not_shared\0"),
            (5, 8, b"file_in_shared.txt\0"),
        ];

        // Only /root/shared is shared, not /root/not_shared
        let registry = ShareRegistry::with_shares(vec![
            make_share("/root/shared", ShareMode::ReadWrite),
        ]);

        let parent_path = Path::new("/root");
        let mut filtered = filter_entries(entries, parent_path, &registry);

        let mut names = Vec::new();
        while let Some(entry) = filtered.next() {
            names.push(entry.name.to_bytes().to_vec());
        }

        // Should include ., .., and shared (which is a share)
        // Should exclude not_shared and file_in_shared.txt (not under any share from /root perspective)
        assert!(names.contains(&b".".to_vec()));
        assert!(names.contains(&b"..".to_vec()));
        assert!(names.contains(&b"shared".to_vec()));
        assert!(!names.contains(&b"not_shared".to_vec()));
        // file_in_shared.txt is not visible because it's not under /root/shared
        // (would need to be /root/shared/file_in_shared.txt)
        assert!(!names.contains(&b"file_in_shared.txt".to_vec()));
    }

    #[test]
    fn test_open_needs_write() {
        // Test the helper function for detecting write access in open flags
        // Note: This would need a MergedPathFs instance, so we test the logic directly

        fn open_needs_write(flags: u32) -> bool {
            let cflags: libc::c_int = flags as libc::c_int;
            if cflags & libc::O_PATH != 0 {
                return false;
            }
            let accmode = cflags & libc::O_ACCMODE;
            accmode == libc::O_WRONLY || accmode == libc::O_RDWR
        }

        // Read-only doesn't need write
        assert!(!open_needs_write(libc::O_RDONLY as u32));

        // Write-only needs write
        assert!(open_needs_write(libc::O_WRONLY as u32));

        // Read-write needs write
        assert!(open_needs_write(libc::O_RDWR as u32));

        // O_PATH doesn't need write even with write flags
        assert!(!open_needs_write((libc::O_PATH | libc::O_RDWR) as u32));
    }

    #[test]
    fn test_create_allowed_child_under_rw_share_in_ancestor_dir() {
        // Scenario: parent=/home/luna is an ancestor (not directly shared),
        // but child=/home/luna/.claude.lock is under an RW share.
        // Creating .claude.lock should be allowed.
        let registry = ShareRegistry::with_shares(vec![
            make_share("/home/luna/.claude.lock", ShareMode::ReadWrite),
        ]);

        let parent = Path::new("/home/luna");
        let child = parent.join(".claude.lock");

        // Parent is not under any share — it's just an ancestor
        assert_eq!(registry.get_permission(parent), None);
        assert!(registry.is_ancestor_of_share(parent));

        // But the child path IS under an RW share
        assert_eq!(
            registry.get_permission(&child),
            Some(ShareMode::ReadWrite)
        );
    }

    #[test]
    fn test_create_blocked_child_under_ro_share_in_ancestor_dir() {
        // Parent is an ancestor, child falls under a read-only share.
        let registry = ShareRegistry::with_shares(vec![
            make_share("/home/luna/.config", ShareMode::ReadOnly),
        ]);

        let parent = Path::new("/home/luna");
        let child = parent.join(".config");

        assert_eq!(registry.get_permission(parent), None);
        assert_eq!(
            registry.get_permission(&child),
            Some(ShareMode::ReadOnly)
        );
    }

    #[test]
    fn test_create_child_not_under_any_share_falls_back_to_parent() {
        // Parent is RW shared, child isn't specifically shared but inherits.
        let registry = ShareRegistry::with_shares(vec![
            make_share("/home/luna/project", ShareMode::ReadWrite),
        ]);

        let parent = Path::new("/home/luna/project");
        let child = parent.join("newfile.txt");

        // Child inherits parent's RW permission
        assert_eq!(
            registry.get_permission(&child),
            Some(ShareMode::ReadWrite)
        );
    }

    #[test]
    fn test_create_child_outside_all_shares_denied() {
        // Parent has no share and is not an ancestor of any share.
        let registry = ShareRegistry::with_shares(vec![
            make_share("/home/luna/project", ShareMode::ReadWrite),
        ]);

        // /etc is completely outside all shares
        assert_eq!(registry.get_permission(Path::new("/etc")), None);
        assert!(!registry.is_ancestor_of_share(Path::new("/etc")));
    }

    #[test]
    fn test_filtered_readdir_virtual_offsets() {
        // Test that FilteredReadDir returns sequential virtual offsets
        let entries = vec![
            CachedDirEntry {
                ino: 1,
                type_: 4,
                name: CString::new(".").unwrap(),
            },
            CachedDirEntry {
                ino: 2,
                type_: 4,
                name: CString::new("..").unwrap(),
            },
            CachedDirEntry {
                ino: 3,
                type_: 4,
                name: CString::new("file1").unwrap(),
            },
            CachedDirEntry {
                ino: 4,
                type_: 4,
                name: CString::new("file2").unwrap(),
            },
        ];

        // Start from offset 0
        let mut iter = FilteredReadDir::from_cache(entries.clone(), 0);
        let e1 = iter.next().unwrap();
        assert_eq!(e1.name.to_bytes(), b".");
        assert_eq!(e1.offset, 1); // Virtual offset

        let e2 = iter.next().unwrap();
        assert_eq!(e2.name.to_bytes(), b"..");
        assert_eq!(e2.offset, 2);

        let e3 = iter.next().unwrap();
        assert_eq!(e3.name.to_bytes(), b"file1");
        assert_eq!(e3.offset, 3);

        let e4 = iter.next().unwrap();
        assert_eq!(e4.name.to_bytes(), b"file2");
        assert_eq!(e4.offset, 4);

        assert!(iter.next().is_none());

        // Test resuming from offset 2 (should start at index 2, i.e. file1)
        let mut iter2 = FilteredReadDir::from_cache(entries, 2);
        let e1 = iter2.next().unwrap();
        assert_eq!(e1.name.to_bytes(), b"file1");
        assert_eq!(e1.offset, 3);

        let e2 = iter2.next().unwrap();
        assert_eq!(e2.name.to_bytes(), b"file2");
        assert_eq!(e2.offset, 4);

        assert!(iter2.next().is_none());
    }
}
