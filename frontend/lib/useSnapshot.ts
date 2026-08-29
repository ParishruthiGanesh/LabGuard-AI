"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import type { ClaimSnapshot } from "./types";

const TERMINAL = new Set(["verdict", "halted_budget", "halted_loop", "halted_approval"]);

/**
 * Polls the claim snapshot.
 *
 * The backend returns the whole claim in one document, so a poll is a single
 * request and the UI never shows a half-updated mix of two states. Polling
 * stops once the claim reaches a terminal state, and pauses while the tab is
 * hidden.
 */
export function useSnapshot(claimId: string | null, intervalMs = 1200) {
  const [snapshot, setSnapshot] = useState<ClaimSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const cancelled = useRef(false);
  const settled = useRef(false);

  const refresh = useCallback(async () => {
    if (!claimId) return null;
    try {
      const next = await api.snapshot(claimId);
      if (!cancelled.current) {
        setSnapshot(next);
        setError(null);
      }
      return next;
    } catch (err) {
      if (!cancelled.current) setError(err instanceof Error ? err.message : String(err));
      return null;
    } finally {
      if (!cancelled.current) setLoading(false);
    }
  }, [claimId]);

  useEffect(() => {
    cancelled.current = false;
    settled.current = false;
    if (!claimId) {
      setLoading(false);
      return;
    }

    const tick = async () => {
      if (document.visibilityState === "hidden") {
        timer.current = setTimeout(tick, intervalMs);
        return;
      }
      const next = await refresh();
      if (cancelled.current) return;
      // Keep polling on error so a restarted backend reconnects on its own.
      // On reaching a terminal state, take one more reading before stopping,
      // so nothing written in the same instant is missed.
      if (next && TERMINAL.has(next.claim.state)) {
        if (settled.current) return;
        settled.current = true;
      }
      timer.current = setTimeout(tick, intervalMs);
    };

    void tick();
    return () => {
      cancelled.current = true;
      if (timer.current) clearTimeout(timer.current);
    };
  }, [claimId, intervalMs, refresh]);

  return { snapshot, error, loading, refresh };
}
