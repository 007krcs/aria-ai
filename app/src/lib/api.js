/**
 * Shared API base URL.
 * - Local dev: empty string — Vite proxy forwards /api/* to localhost:8000
 * - Production: set VITE_API_URL=https://your-backend.railway.app
 */
export const API_BASE = import.meta.env.VITE_API_URL || "";
