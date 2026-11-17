import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { qacTiming, serpTiming, useQac, useSearch } from "./search.js";

// A fake fetch whose responses resolve when the test says so, so responses can arrive out of order.
function fakeFetch() {
  const calls = [];
  const fetch = vi.fn((url) => new Promise((resolve) => calls.push({ url: String(url), resolve })));
  const reply = (i, body, status = 200) =>
    act(async () => calls[i].resolve({ ok: status < 400, status, json: async () => body, text: async () => JSON.stringify(body) }));
  return { fetch, calls, reply };
}

const qacBody = (prefix, l2 = { status: "ok", items: [] }) => ({
  prefix, l1: [{ completion: `${prefix}x` }], l2, request_id: `r-${prefix}`, timings_ms: { l1: 2, l2: 30 },
});
const serpBody = (query) => ({
  query, hits: 1, l1: [{ product_id: "B1" }], l2: { status: "ok", items: [{ product_id: "B1" }] },
  embedding: "custom", query_type: "HYBRID", timings_ms: { embed: 50, search: 200, l2: 90 },
});

let f;
beforeEach(() => {
  f = fakeFetch();
  vi.stubGlobal("fetch", f.fetch);
});
afterEach(() => vi.unstubAllGlobals());

describe("timings", () => {
  it("splits a keystroke into debounce, network + app, server and render", () => {
    const t = qacTiming(qacBody("ca"), { pressed: 0, sent: 20, received: 100 }, 110);
    expect(t).toMatchObject({ debounce: 20, server: 32, network: 48, render: 10, total: 110, l1: 2, l2: 30,
                              status: "ok", request_id: "r-ca", prefix: "ca" });
  });

  it("never shows a negative network time when the server's clock says more than the round trip", () => {
    expect(qacTiming(qacBody("ca"), { pressed: 0, sent: 20, received: 40 }, 45).network).toBe(0);
    expect(serpTiming(serpBody("bag"), { sent: 0, received: 300 }, 310).network).toBe(0);
  });

  it("times a search from Enter, with no debounce", () => {
    expect(serpTiming(serpBody("bag"), { sent: 0, received: 400 }, 420)).toMatchObject({
      server: 340, network: 60, render: 20, total: 420, embed: 50, search: 200, l2: 90, hits: 1, status: "ok" });
  });
});

describe("useQac", () => {
  const keyAt = () => ({ current: performance.now() });

  it("fetches after the debounce with the session and user, then times the keystroke", async () => {
    const { result } = renderHook(() => useQac({ q: "ca", sessionId: "s1", userId: "u1", debounceMs: 0, keyAt: keyAt() }));
    await waitFor(() => expect(f.calls).toHaveLength(1));
    expect(f.calls[0].url).toBe("/api/qac?q=ca&session_id=s1&user_id=u1");
    await f.reply(0, qacBody("ca"));
    expect(result.current.result.prefix).toBe("ca");
    await waitFor(() => expect(result.current.e2e?.prefix).toBe("ca"));
  });

  it("drops a response to an older keystroke", async () => {
    const ref = keyAt();
    const { result, rerender } = renderHook(({ q }) => useQac({ q, debounceMs: 0, keyAt: ref }), { initialProps: { q: "c" } });
    await waitFor(() => expect(f.calls).toHaveLength(1));
    rerender({ q: "ca" });
    await waitFor(() => expect(f.calls).toHaveLength(2));
    await f.reply(1, qacBody("ca"));
    await f.reply(0, qacBody("c"));
    expect(result.current.result.prefix).toBe("ca");
  });

  it("shows a failed keystroke's error until the next one succeeds", async () => {
    const ref = keyAt();
    const { result, rerender } = renderHook(({ q }) => useQac({ q, debounceMs: 0, keyAt: ref }), { initialProps: { q: "c" } });
    await waitFor(() => expect(f.calls).toHaveLength(1));
    await f.reply(0, { detail: "boom" }, 500);
    expect(result.current.error).toMatch(/^Suggestions failed: /);
    expect(result.current.result).toBeNull(); // the error body is never treated as completions
    rerender({ q: "ca" });
    await waitFor(() => expect(f.calls).toHaveLength(2));
    await f.reply(1, qacBody("ca"));
    expect(result.current.error).toBeNull();
    expect(result.current.result.prefix).toBe("ca");
  });

  it("doesn't time a refetch that no keystroke caused", async () => {
    const { result } = renderHook(() => useQac({ q: "ca", debounceMs: 0, keyAt: { current: null } }));
    await waitFor(() => expect(f.calls).toHaveLength(1));
    await f.reply(0, qacBody("ca"));
    expect(result.current.result.prefix).toBe("ca");
    await new Promise((r) => setTimeout(r, 50));
    expect(result.current.e2e).toBeNull();
  });
});

describe("useSearch", () => {
  it("searches with the session and user, then times the search", async () => {
    const { result } = renderHook(() => useSearch({ sessionId: "s1", userId: "u1" }));
    act(() => result.current.run("bag"));
    expect(result.current.searching).toBe(true);
    expect(f.calls[0].url).toBe("/api/search?q=bag&session_id=s1&user_id=u1");
    await f.reply(0, serpBody("bag"));
    expect(result.current.serp.query).toBe("bag");
    expect(result.current.searching).toBe(false);
    await waitFor(() => expect(result.current.e2e?.hits).toBe(1));
  });

  it("ignores blank searches", () => {
    const { result } = renderHook(() => useSearch({}));
    act(() => result.current.run("  "));
    expect(f.calls).toHaveLength(0);
  });

  it("keeps the newest search when an older one answers last", async () => {
    const { result } = renderHook(() => useSearch({}));
    act(() => result.current.run("a"));
    act(() => result.current.run("bag"));
    await f.reply(1, serpBody("bag"));
    await f.reply(0, serpBody("a"));
    expect(result.current.serp.query).toBe("bag");
  });

  it("reports a failed search, and the next search clears it", async () => {
    const { result } = renderHook(() => useSearch({}));
    act(() => result.current.run("bag"));
    await f.reply(0, { detail: "boom" }, 500);
    expect(result.current.error).toMatch(/^Search failed: /);
    expect(result.current.searching).toBe(false);
    act(() => result.current.run("bag"));
    expect(result.current.error).toBeNull();
  });

  it("clear drops the results and any search still in flight", async () => {
    const { result } = renderHook(() => useSearch({}));
    act(() => result.current.run("bag"));
    act(() => result.current.clear());
    await f.reply(0, serpBody("bag"));
    expect(result.current.serp).toBeNull();
    expect(result.current.searching).toBe(false);
  });

  it("re-runs the current search when the user signs in or out, as the new user", async () => {
    const { result, rerender } = renderHook(({ userId }) => useSearch({ userId }), { initialProps: { userId: null } });
    act(() => result.current.run("bag"));
    await f.reply(0, serpBody("bag"));
    rerender({ userId: "u1" });
    await waitFor(() => expect(f.calls).toHaveLength(2));
    expect(f.calls[1].url).toBe("/api/search?q=bag&user_id=u1");
  });

  it("re-runs the current search when the clickstream is cleared, but not when one is played", async () => {
    const { result, rerender } = renderHook(({ sessionId }) => useSearch({ sessionId }),
                                            { initialProps: { sessionId: null } });
    act(() => result.current.run("bag"));
    await f.reply(0, serpBody("bag"));
    rerender({ sessionId: "s1" });
    await new Promise((r) => setTimeout(r, 20));
    expect(f.calls).toHaveLength(1); // a new session waits for the next search
    rerender({ sessionId: null });
    await waitFor(() => expect(f.calls).toHaveLength(2));
    expect(f.calls[1].url).toBe("/api/search?q=bag");
  });

  it("doesn't search on sign-in before anything was searched", async () => {
    const { rerender } = renderHook(({ userId }) => useSearch({ userId }), { initialProps: { userId: null } });
    rerender({ userId: "u1" });
    await new Promise((r) => setTimeout(r, 20));
    expect(f.calls).toHaveLength(0);
  });
});
