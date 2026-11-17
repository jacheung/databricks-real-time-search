// QAC and SERP requests from the browser: each hook owns its fetch, drops responses that a newer request has
// superseded, keeps its own error, and times the request until its results are on screen.
import { useEffect, useRef, useState } from "react";

// Keystroke → on screen. Browser clocks (performance.now) except L1 and L2, the response's own timings; "network" is
// the rest of the round trip (proxy, TLS, FastAPI, JSON), never below 0 since app and browser clocks differ.
export function qacTiming(res, { pressed, sent, received }, painted) {
  const t = res.timings_ms;
  const server = t.l1 + (t.l2 ?? 0);
  return { debounce: sent - pressed, network: Math.max(0, received - sent - server), server,
           render: painted - received, total: painted - pressed,
           l1: t.l1, l2: t.l2, status: res.l2?.status, request_id: res.request_id, prefix: res.prefix };
}

// Search → results on screen: no debounce, the search starts on the Enter or pick itself.
export function serpTiming(res, { sent, received }, painted) {
  const t = res.timings_ms;
  const server = t.embed + t.search + (t.l2 ?? 0);
  return { server, network: Math.max(0, received - sent - server), embed: t.embed, search: t.search, l2: t.l2,
           render: painted - received, total: painted - sent, status: res.l2?.status, hits: res.hits,
           embedding: res.embedding };
}

function getJson(url) {
  return fetch(url).then((r) => (r.ok ? r.json() : r.text().then((t) => Promise.reject(new Error(t)))));
}

function query(params) {
  const p = new URLSearchParams(params.q === undefined ? {} : { q: params.q });
  if (params.sessionId) p.set("session_id", params.sessionId);
  if (params.userId) p.set("user_id", params.userId);
  return p;
}

// Two frames after the response is in state, the results are painted: stop the clock there.
function useTimed(response, clock, timing) {
  const [e2e, setE2e] = useState(null);
  useEffect(() => {
    if (!response || !clock) return;
    let frame = requestAnimationFrame(() => {
      frame = requestAnimationFrame(() => setE2e(timing(response, clock, performance.now())));
    });
    return () => cancelAnimationFrame(frame);
  }, [response]);
  return [e2e, setE2e];
}

// Completions for what's typed, `debounceMs` after the last keystroke. `keyAt` holds the keystroke's event time
// (performance.now's clock) and is consumed here: a refetch with no keystroke behind it (a new session or sign-in)
// isn't timed.
export function useQac({ q, sessionId, userId, debounceMs, keyAt }) {
  const [state, setState] = useState({ result: null, clock: null, error: null });
  const latest = useRef(0);
  useEffect(() => {
    const seq = ++latest.current;
    const pressed = keyAt.current;
    keyAt.current = null;
    const t = setTimeout(() => {
      const sent = performance.now();
      getJson(`/api/qac?${query({ q, sessionId, userId })}`)
        .then((result) => seq === latest.current && setState({
          result, error: null, clock: pressed === null ? null : { pressed, sent, received: performance.now() } }))
        .catch((e) => seq === latest.current && setState((s) => ({ ...s, error: `Suggestions failed: ${e.message}` })));
    }, debounceMs);
    return () => clearTimeout(t);
  }, [q, sessionId, userId, debounceMs]);
  const [e2e] = useTimed(state.result, state.clock, qacTiming);
  return { result: state.result, e2e, error: state.error };
}

// SERP for `run(text)`, with the current session and user so L2 sees the same features QAC did. Signing in or out
// re-runs the current search, as a store's reload after login would; so does clearing the clickstream, so its L2
// order stops using the old session. Both are timed like any search: no keystroke or debounce is involved.
export function useSearch({ sessionId, userId }) {
  const [state, setState] = useState({ serp: null, clock: null, searching: false, error: null });
  const latest = useRef(0);
  const current = useRef({ sessionId, userId, serp: null });
  current.current = { sessionId, userId, serp: state.serp };

  function run(text) {
    if (!text.trim()) return;
    const seq = ++latest.current;
    const { sessionId, userId } = current.current;
    setState((s) => ({ ...s, searching: true, error: null }));
    const sent = performance.now();
    getJson(`/api/search?${query({ q: text, sessionId, userId })}`)
      .then((serp) => seq === latest.current && setState({
        serp, clock: { sent, received: performance.now() }, searching: false, error: null }))
      .catch((e) => seq === latest.current && setState((s) => ({
        ...s, searching: false, error: `Search failed: ${e.message}` })));
  }

  function clear() {
    latest.current++; // drop an in-flight search
    setState({ serp: null, clock: null, searching: false, error: null });
    setE2e(null);
  }

  const mounted = useRef(false);
  useEffect(() => {
    if (mounted.current && current.current.serp) run(current.current.serp.query);
  }, [userId]);
  const hadSession = useRef(false);
  useEffect(() => {
    if (hadSession.current && !sessionId && current.current.serp) run(current.current.serp.query);
    hadSession.current = !!sessionId;
  }, [sessionId]);
  useEffect(() => {
    mounted.current = true;
  }, []);

  const [e2e, setE2e] = useTimed(state.serp, state.clock, serpTiming);
  return { serp: state.serp, e2e, searching: state.searching, error: state.error, run, clear };
}
