// Copyright 2026 The Virtiofs Project Developers
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

//! Token-based authentication for HTTP APIs.

use axum::{
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::Response,
};

/// Extract bearer token from Authorization header.
fn extract_bearer_token(request: &Request) -> Option<&str> {
    request
        .headers()
        .get("Authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
}

/// Create an authentication middleware for the given token.
pub fn create_auth_middleware(
    expected_token: Option<String>,
) -> impl Fn(Request, Next) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Response, StatusCode>> + Send>> + Clone {
    move |request: Request, next: Next| {
        let expected_token = expected_token.clone();
        Box::pin(async move {
            // If no token is configured, allow all requests
            let Some(expected) = expected_token else {
                return Ok(next.run(request).await);
            };

            // Extract and verify the token
            let Some(provided) = extract_bearer_token(&request) else {
                return Err(StatusCode::UNAUTHORIZED);
            };

            if provided != expected {
                return Err(StatusCode::UNAUTHORIZED);
            }

            Ok(next.run(request).await)
        })
    }
}

/// Middleware layer for Admin API authentication.
#[derive(Clone)]
pub struct AdminAuth {
    pub token: Option<String>,
}

/// Middleware layer for VM API authentication.
#[derive(Clone)]
pub struct VmAuth {
    pub token: Option<String>,
}

impl AdminAuth {
    pub fn new(token: Option<String>) -> Self {
        AdminAuth { token }
    }
}

impl VmAuth {
    pub fn new(token: Option<String>) -> Self {
        VmAuth { token }
    }
}
