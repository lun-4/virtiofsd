// Copyright 2026 The Virtiofs Project Developers
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

//! Admin API for managing shares.
//!
//! Endpoints:
//! - GET /shares - List all shares
//! - POST /shares - Add a new share
//! - DELETE /shares?path=<path> - Remove a share
//! - GET /pending-requests - List VM requests
//! - POST /pending-requests/{id}/approve - Approve a request
//! - POST /pending-requests/{id}/deny - Deny a request

use super::{HttpState, RequestStatus, ShareRequest};
use crate::share::{Share, ShareMode};
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    middleware,
    routing::{delete, get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;

/// Request body for adding a share.
#[derive(Debug, Deserialize)]
pub struct AddShareRequest {
    pub path: String,
    pub mode: String,
}

/// Response for share operations.
#[derive(Debug, Serialize)]
pub struct ShareResponse {
    pub id: u64,
    pub path: String,
    pub mode: String,
}

/// Query parameters for removing a share.
#[derive(Debug, Deserialize)]
pub struct RemoveShareQuery {
    pub path: String,
}

/// List all shares.
async fn list_shares(State(state): State<Arc<HttpState>>) -> Json<Vec<ShareResponse>> {
    let shares = state.registry.list_shares();
    let response: Vec<ShareResponse> = shares
        .into_iter()
        .map(|entry| ShareResponse {
            id: entry.id,
            path: entry.share.path().display().to_string(),
            mode: entry.share.mode().as_short_str().to_string(),
        })
        .collect();
    Json(response)
}

/// Add a new share.
async fn add_share(
    State(state): State<Arc<HttpState>>,
    Json(req): Json<AddShareRequest>,
) -> Result<Json<ShareResponse>, (StatusCode, String)> {
    // Parse mode
    let mode: ShareMode = req
        .mode
        .parse()
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid mode: {}", e)))?;

    // Create share
    let share = Share::new(&req.path, mode)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid path: {}", e)))?;

    // Add to registry
    let id = state.registry.add_share(share.clone());

    log::info!("Added share: {} ({})", req.path, req.mode);

    Ok(Json(ShareResponse {
        id,
        path: share.path().display().to_string(),
        mode: share.mode().as_short_str().to_string(),
    }))
}

/// Remove a share by path.
async fn remove_share(
    State(state): State<Arc<HttpState>>,
    Query(query): Query<RemoveShareQuery>,
) -> Result<StatusCode, (StatusCode, String)> {
    let path = PathBuf::from(&query.path);
    match state.registry.remove_share(&path) {
        Some(_) => {
            log::info!("Removed share: {}", query.path);
            Ok(StatusCode::NO_CONTENT)
        }
        None => Err((
            StatusCode::NOT_FOUND,
            format!("Share not found: {}", query.path),
        )),
    }
}

/// List pending requests from VM.
async fn list_pending_requests(State(state): State<Arc<HttpState>>) -> Json<Vec<ShareRequest>> {
    let requests = state.pending_requests.read().await;
    Json(requests.clone())
}

/// Approve a pending request.
async fn approve_request(
    State(state): State<Arc<HttpState>>,
    Path(id): Path<u64>,
) -> Result<Json<ShareResponse>, (StatusCode, String)> {
    let mut requests = state.pending_requests.write().await;

    // Find the request
    let request = requests
        .iter_mut()
        .find(|r| r.id == id)
        .ok_or((StatusCode::NOT_FOUND, "Request not found".to_string()))?;

    if request.status != RequestStatus::Pending {
        return Err((
            StatusCode::BAD_REQUEST,
            "Request already processed".to_string(),
        ));
    }

    // Parse mode
    let mode: ShareMode = request
        .mode
        .parse()
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid mode: {}", e)))?;

    // Create and add share
    let share = Share::new(&request.path, mode)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid path: {}", e)))?;

    let share_id = state.registry.add_share(share.clone());

    // Mark as approved
    request.status = RequestStatus::Approved;

    log::info!(
        "Approved share request {}: {} ({})",
        id,
        request.path,
        request.mode
    );

    Ok(Json(ShareResponse {
        id: share_id,
        path: share.path().display().to_string(),
        mode: share.mode().as_short_str().to_string(),
    }))
}

/// Deny a pending request.
async fn deny_request(
    State(state): State<Arc<HttpState>>,
    Path(id): Path<u64>,
) -> Result<StatusCode, (StatusCode, String)> {
    let mut requests = state.pending_requests.write().await;

    // Find the request
    let request = requests
        .iter_mut()
        .find(|r| r.id == id)
        .ok_or((StatusCode::NOT_FOUND, "Request not found".to_string()))?;

    if request.status != RequestStatus::Pending {
        return Err((
            StatusCode::BAD_REQUEST,
            "Request already processed".to_string(),
        ));
    }

    request.status = RequestStatus::Denied;

    log::info!("Denied share request {}: {}", id, request.path);

    Ok(StatusCode::NO_CONTENT)
}

/// Check bearer token authentication.
async fn check_auth(
    expected_token: String,
    req: axum::extract::Request,
    next: middleware::Next,
) -> Result<axum::response::Response, StatusCode> {
    if let Some(auth) = req.headers().get("Authorization") {
        if let Ok(auth_str) = auth.to_str() {
            if let Some(provided) = auth_str.strip_prefix("Bearer ") {
                if provided == expected_token {
                    return Ok(next.run(req).await);
                }
            }
        }
    }
    Err(StatusCode::UNAUTHORIZED)
}

/// Create the Admin API router.
pub fn create_router(state: Arc<HttpState>, token: Option<String>) -> Router {
    let router = Router::new()
        .route("/shares", get(list_shares).post(add_share))
        .route("/shares", delete(remove_share))
        .route("/pending-requests", get(list_pending_requests))
        .route("/pending-requests/{id}/approve", post(approve_request))
        .route("/pending-requests/{id}/deny", post(deny_request))
        .with_state(state);

    // Add authentication middleware if token is configured
    if let Some(expected_token) = token {
        router.layer(middleware::from_fn(move |req, next| {
            check_auth(expected_token.clone(), req, next)
        }))
    } else {
        router
    }
}

/// Admin API type for module exports.
pub struct AdminApi;
