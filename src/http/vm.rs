// Copyright 2026 The Virtiofs Project Developers
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

//! VM API for requesting shares.
//!
//! Endpoints:
//! - POST /request-share - Request a new share
//! - GET /request-status/{id} - Check request status

use super::{HttpState, RequestStatus, ShareRequest};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    middleware,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Request body for requesting a share.
#[derive(Debug, Deserialize)]
pub struct RequestShareBody {
    pub path: String,
    pub mode: String,
}

/// Response for share request creation.
#[derive(Debug, Serialize)]
pub struct RequestShareResponse {
    pub id: u64,
    pub status: RequestStatus,
    pub message: String,
}

/// Response for request status check.
#[derive(Debug, Serialize)]
pub struct RequestStatusResponse {
    pub id: u64,
    pub path: String,
    pub mode: String,
    pub status: RequestStatus,
}

/// Request a new share.
async fn request_share(
    State(state): State<Arc<HttpState>>,
    Json(req): Json<RequestShareBody>,
) -> Result<Json<RequestShareResponse>, (StatusCode, String)> {
    // Validate mode
    if req.mode != "ro" && req.mode != "rw" && req.mode != "readonly" && req.mode != "readwrite" {
        return Err((
            StatusCode::BAD_REQUEST,
            "Invalid mode: must be 'ro' or 'rw'".to_string(),
        ));
    }

    // Normalize mode
    let mode = if req.mode == "readonly" { "ro" } else if req.mode == "readwrite" { "rw" } else { &req.mode };

    // Create request
    let id = state.next_request_id();
    let request = ShareRequest {
        id,
        path: req.path.clone(),
        mode: mode.to_string(),
        status: RequestStatus::Pending,
    };

    // Add to pending requests
    state.pending_requests.write().await.push(request);

    log::info!(
        "VM requested share: {} ({}) - request ID {}",
        req.path,
        mode,
        id
    );

    Ok(Json(RequestShareResponse {
        id,
        status: RequestStatus::Pending,
        message: "Request submitted, awaiting admin approval".to_string(),
    }))
}

/// Check request status.
async fn get_request_status(
    State(state): State<Arc<HttpState>>,
    Path(id): Path<u64>,
) -> Result<Json<RequestStatusResponse>, (StatusCode, String)> {
    let requests = state.pending_requests.read().await;

    let request = requests
        .iter()
        .find(|r| r.id == id)
        .ok_or((StatusCode::NOT_FOUND, "Request not found".to_string()))?;

    Ok(Json(RequestStatusResponse {
        id: request.id,
        path: request.path.clone(),
        mode: request.mode.clone(),
        status: request.status,
    }))
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

/// Create the VM API router.
pub fn create_router(state: Arc<HttpState>, token: Option<String>) -> Router {
    let router = Router::new()
        .route("/request-share", post(request_share))
        .route("/request-status/{id}", get(get_request_status))
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

/// VM API type for module exports.
pub struct VmApi;
