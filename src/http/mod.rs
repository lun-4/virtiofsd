// Copyright 2026 The Virtiofs Project Developers
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

//! HTTP control plane for dynamic share management.
//!
//! This module provides HTTP APIs for:
//! - Admin API: Managing shares, approving/denying requests
//! - VM API: Requesting new shares from the guest

pub mod admin;
pub mod auth;
pub mod vm;

use crate::share_registry::ShareRegistry;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;

pub use admin::AdminApi;
pub use auth::{AdminAuth, VmAuth};
pub use vm::VmApi;

/// Configuration for the HTTP control plane.
#[derive(Debug, Clone)]
pub struct HttpConfig {
    /// Port for the VM API.
    pub vm_api_port: u16,
    /// Port for the Admin API.
    pub admin_api_port: u16,
    /// Token for VM API authentication.
    pub vm_token: Option<String>,
    /// Token for Admin API authentication.
    pub admin_token: Option<String>,
    /// VM name for desktop notifications.
    pub vm_name: Option<String>,
}

/// Pending share request from VM.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ShareRequest {
    /// Unique request ID.
    pub id: u64,
    /// Requested path.
    pub path: String,
    /// Requested access mode.
    pub mode: String,
    /// Request status: pending, approved, denied.
    pub status: RequestStatus,
    /// Reason for denial (if denied).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deny_reason: Option<String>,
}

/// Status of a share request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RequestStatus {
    Pending,
    Approved,
    Denied,
}

/// Shared state for HTTP APIs.
pub struct HttpState {
    /// Share registry.
    pub registry: Arc<ShareRegistry>,
    /// Pending requests from VM.
    pub pending_requests: RwLock<Vec<ShareRequest>>,
    /// Next request ID.
    next_request_id: std::sync::atomic::AtomicU64,
}

impl HttpState {
    /// Create new HTTP state with the given share registry.
    pub fn new(registry: Arc<ShareRegistry>) -> Self {
        HttpState {
            registry,
            pending_requests: RwLock::new(Vec::new()),
            next_request_id: std::sync::atomic::AtomicU64::new(1),
        }
    }

    /// Generate a new request ID.
    pub fn next_request_id(&self) -> u64 {
        self.next_request_id
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
    }
}

/// Start the HTTP control plane servers.
///
/// This function spawns two HTTP servers:
/// - Admin API on `admin_port`
/// - VM API on `vm_port`
///
/// Returns handles that can be used to gracefully shut down the servers.
pub async fn start_http_servers(
    config: HttpConfig,
    registry: Arc<ShareRegistry>,
) -> std::io::Result<()> {
    let state = Arc::new(HttpState::new(registry));

    // Create Admin API router
    let admin_router = admin::create_router(Arc::clone(&state), config.admin_token.clone());
    let admin_addr = SocketAddr::from(([0, 0, 0, 0], config.admin_api_port));

    // Create VM API router
    let vm_router = vm::create_router(Arc::clone(&state), config.vm_token.clone(), config.vm_name.clone());
    let vm_addr = SocketAddr::from(([0, 0, 0, 0], config.vm_api_port));

    log::info!("Starting Admin API on {}", admin_addr);
    log::info!("Starting VM API on {}", vm_addr);

    // Spawn both servers
    let admin_handle = tokio::spawn(async move {
        let listener = tokio::net::TcpListener::bind(admin_addr).await?;
        axum::serve(listener, admin_router).await
    });

    let vm_handle = tokio::spawn(async move {
        let listener = tokio::net::TcpListener::bind(vm_addr).await?;
        axum::serve(listener, vm_router).await
    });

    // Wait for both servers (they run indefinitely unless an error occurs)
    tokio::select! {
        result = admin_handle => {
            if let Err(e) = result {
                log::error!("Admin API server error: {}", e);
            }
        }
        result = vm_handle => {
            if let Err(e) = result {
                log::error!("VM API server error: {}", e);
            }
        }
    }

    Ok(())
}
